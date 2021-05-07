/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
import Scene.RaytracingInline;
import Scene.Scene;
import Scene.Shading;
import Utils.Helpers;
import Utils.Math.HashUtils;
import Utils.Color.ColorMap;
import Utils.Color.ColorHelpers;
import SharedTypes;

struct InteractionPass
{
    InteractionPassParams params;

    RWStructuredBuffer<PixelData> pixelData;

    /** Run interaction pass for the given pixel.
    */
    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        // Initialize pixel data for the selected pixel.
        if (all(pixel == params.selectedPixel))
        {
            pixelData[0].meshInstanceID = PixelData::kInvalidID;
            pixelData[0].meshID = PixelData::kInvalidID;
            pixelData[0].materialID = PixelData::kInvalidID;
            pixelData[0].curveInstanceID = PixelData::kInvalidID;
            pixelData[0].curveID = PixelData::kInvalidID;
        }

        // Trace primary ray.
        RayDesc ray = gScene.camera.computeRayPinhole(pixel, params.frameDim).toRayDesc();

        RayQuery<RAY_FLAG_NONE> rayQuery;
        //RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rayQuery;

        rayQuery.TraceRayInline(
            gRtScene,
            RAY_FLAG_NONE, // OR'd with template flags above
            0xff, // InstanceInclusionMask
            ray);

        float2 curveCommittedAttribs = {};

        while (rayQuery.Proceed())
        {
            switch (rayQuery.CandidateType())
            {
            case CANDIDATE_NON_OPAQUE_TRIANGLE:
            {
                // Alpha test for non-opaque geometry.
                HitInfo hit = getCandidateTriangleHit(rayQuery);
                VertexData v = gScene.getVertexData(hit);
                uint materialID = gScene.getMaterialID(hit.instanceID);

                if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) continue;

                rayQuery.CommitNonOpaqueTriangleHit();
                break;
            }
            case CANDIDATE_PROCEDURAL_PRIMITIVE:
            {
                float t;
                HitInfo hit;
                bool valid = getCandidateCurveHit(rayQuery, ray, t, hit);

                if (valid)
                {
                    rayQuery.CommitProceduralPrimitiveHit(t);
                    curveCommittedAttribs = hit.barycentrics;
                }
                break;
            }
            }
        }

        // Process hit/miss.
        float hitT = 1e30;
        switch (rayQuery.CommittedStatus())
        {
        case COMMITTED_TRIANGLE_HIT:
        {
            HitInfo hit = getCommittedTriangleHit(rayQuery);
            hitT = rayQuery.CommittedRayT();
            handleHit(pixel, ray.Direction, hit);
            break;
        }
        case COMMITTED_PROCEDURAL_PRIMITIVE_HIT:
        {
            HitInfo hit = getCommittedCurveHit(rayQuery, curveCommittedAttribs);
            hitT = rayQuery.CommittedRayT();
            const float3 curveHitPosW = ray.Origin + ray.Direction * hitT;
            handleHit(pixel, ray.Direction, hit, curveHitPosW);
            break;
        }
        default: // COMMITTED_NOTHING
        {
            ;
        }
        }
    }

    void handleHit(const uint2 pixel, const float3 dir, const HitInfo hit, const float3 curveHitPosW = {})
    {
        uint materialID = PixelData::kInvalidID;

        uint meshID = PixelData::kInvalidID;
        uint meshInstanceID = PixelData::kInvalidID;

        uint curveID = PixelData::kInvalidID;
        uint curveInstanceID = PixelData::kInvalidID;

        VertexData v;
        ShadingData sd;

        switch (hit.getType())
        {
        case InstanceType::TriangleMesh:
        {
            meshInstanceID = hit.instanceID;
            meshID = gScene.getMeshInstance(meshInstanceID).meshID;
            materialID = gScene.getMaterialID(meshInstanceID);

            break;
        }
        case InstanceType::Curve:
        {
            curveInstanceID = hit.instanceID;
            curveID = gScene.getCurveInstance(curveInstanceID).curveID;
            materialID = gScene.getCurveMaterialID(curveInstanceID);

            break;
        }
        default:
            // Should not happen.
            return;
        }

        // Write pixel data for the selected pixel.
        if (all(pixel == params.selectedPixel))
        {
            pixelData[0].meshInstanceID = meshInstanceID;
            pixelData[0].meshID = meshID;
            pixelData[0].materialID = materialID;
            pixelData[0].curveInstanceID = curveInstanceID;
            pixelData[0].curveID = curveID;
        }
    }
};

cbuffer CB
{
    InteractionPass gInteractionPass;
}

/** Compute shader entry point for interaction pass.
*/
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gInteractionPass.execute(dispatchThreadId.xy);
}
