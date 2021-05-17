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
#include "InteractionPass.h"


namespace
{
    const char kDesc[] = "Enables per frame user Interaction.";
    const char out[] = "dst";
    const char in[] = "src";
    const char kShaderFile[] = "RenderPasses/InteractionPass/InteractionPass.cs.slang";
    const char kShaderModel[] = "6_5";

    const std::string kOutput = "output";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("InteractionPass", kDesc, InteractionPass::create);
}

InteractionPass::SharedPtr InteractionPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new InteractionPass);
    return pPass;
}

std::string InteractionPass::getDesc() { return kDesc; }

Dictionary InteractionPass::getScriptingDictionary()
{
    return Dictionary();
}

InteractionPass::InteractionPass()
{
    if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw std::exception("Raytracing Tier 1.1 is not supported by the current device");
    }

    Program::Desc desc;
    desc.addShaderLibrary(kShaderFile).csEntry("main").setShaderModel(kShaderModel);
    mpInteractionPass = ComputePass::create(desc, Program::DefineList(), false);
    mpFence = GpuFence::create();
}

RenderPassReflection InteractionPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput(out, "The destination texture");
    r.addInput(in, "The source texture");
    return r;
}

void InteractionPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;

    if (mpScene)
    {
        // Prepare our programs for the scene.
        Shader::DefineList defines = mpScene->getSceneDefines();

        // Disable discard and gradient operations.
        defines.add("_MS_DISABLE_ALPHA_TEST");
        defines.add("_DEFAULT_ALPHA_TEST");

        mpInteractionPass->getProgram()->addDefines(defines);
        mpInteractionPass->setVars(nullptr); // Trigger recompile

        // Bind variables.
        auto var = mpInteractionPass->getRootVar()["CB"]["gInteractionPass"];
        if (!mpPixelDataBuffer)
        {
            mpPixelDataBuffer = Buffer::createStructured(var["pixelData"], 1, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            mpPixelDataStaging = Buffer::createStructured(var["pixelData"], 1, ResourceBindFlags::None, Buffer::CpuAccess::Read, nullptr, false);
        }
        var["pixelData"] = mpPixelDataBuffer;
    }
}

void InteractionPass::compile(RenderContext* pContext, const CompileData& compileData)
{
    //pFrameDim = compileData.defaultTexDims;
    mParams.frameDim = compileData.defaultTexDims;
}

void InteractionPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mpScene)
    {

        mpScene->setRaytracingShaderData(pRenderContext, mpInteractionPass->getRootVar());

        ShaderVar var = mpInteractionPass->getRootVar()["CB"]["gInteractionPass"];
        var["params"].setBlob(mParams);

        //mpInteractionPass->execute(pRenderContext, uint3(pFrameDim, 1));
        mpInteractionPass->execute(pRenderContext, uint3(mParams.frameDim, 1));

        pRenderContext->copyResource(mpPixelDataStaging.get(), mpPixelDataBuffer.get());
        pRenderContext->flush(false);
        mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

        if (mRightMouseClicked) {
            assert(mpPixelDataStaging);
            mpFence->syncCpu();
            mpPixelData = *reinterpret_cast<const PixelData*>(mpPixelDataStaging->map(Buffer::MapType::Read));

            if (mpPixelData.meshID != PixelData::kInvalidID)
            {

                glm::mat4 transform = mpScene->getAnimationController()->getGlobalMatrices()[mpScene->getMeshInstance(mpPixelData.meshInstanceID).globalMatrixID];

                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        std::cout << transform[i][j] << " ";
                    }
                    std::cout << std::endl;
                }
                mTranslation = transform[3].xyz;
                mScaling = glm::vec3(glm::length(transform[0]), glm::length(transform[1]), glm::length(transform[2]));
                auto roti = glm::mat4(transform[0] / mScaling[0], transform[1] / mScaling[1], transform[2] / mScaling[2], transform[3]);
                mRotation = glm::eulerAngles(glm::quat_cast(roti));
            }

            mRightMouseClicked = false;
            mPixelDataAvailable = true;
        }

        mParams.frameCount++;
    }

    // Copy rendered input to output
    const auto& pSrcTex = renderData[in]->asTexture();
    const auto& pDstTex = renderData[out]->asTexture();

    if (pSrcTex && pDstTex)
    {
        pRenderContext->blit(pSrcTex->getSRV(), pDstTex->getRTV());
    }
    else
    {
        logWarning("InteractionPass::execute() - missing an input or output resource");
    }
}

bool InteractionPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::RightButtonDown)
    {
        float2 cursorPos = mouseEvent.pos * (float2)mParams.frameDim;
        mParams.selectedPixel = (uint2)glm::clamp(cursorPos, float2(0.f), float2(mParams.frameDim.x - 1, mParams.frameDim.y - 1));
        mRightMouseClicked = true;
    }

    return false;
}

void InteractionPass::renderUI(Gui::Widgets& widget)
{
    widget.var("Selected pixel", mParams.selectedPixel);

    if (mPixelDataAvailable)
    {
        std::ostringstream oss;
        if (mpPixelData.meshID != PixelData::kInvalidID)
        {
            uint matID = mpScene->getMeshInstance(mpPixelData.meshInstanceID).globalMatrixID;
            glm::mat4 transform = mpScene->getAnimationController()->getGlobalMatrices()[matID];
            oss << "Selected Mesh:" << std::endl
                << "Mesh ID: " << mpPixelData.meshID << std::endl
                << "Mesh name: " << (mpScene->hasMesh(mpPixelData.meshID) ? mpScene->getMeshName(mpPixelData.meshID) : "unknown") << std::endl
                << "Mesh instance ID: " << mpPixelData.meshInstanceID << std::endl
                << "Matrix ID: " << matID << std::endl
                << "Material ID: " << mpPixelData.materialID << std::endl
                << "Num Mats: " << mpScene->getAnimationController()->getGlobalMatrices().size() << std::endl;

            Falcor::Animation::SharedPtr ptr = Falcor::Animation::create("interaction_hack", matID, 0.0);
            Falcor::Animation::Keyframe kf;
            kf.rotation = glm::quat(mRotation);
            kf.scaling = mScaling;
            kf.translation = mTranslation;
            kf.time = 0.0f;
            ptr->addKeyframe(kf);
            ptr->setPostInfinityBehavior(Falcor::Animation::Behavior::Constant);

            for (auto& anim : mpScene->getAnimations())
            {
                if (anim->getName() == "interaction_hack")
                {
                    anim = ptr;
                    ptr.reset();
                    break;
                }
            }

            if (ptr)
                mpScene->getAnimations().push_back(ptr);
        }
        else if (mpPixelData.curveInstanceID != PixelData::kInvalidID)
        {
            oss << "Curve ID: " << mpPixelData.curveID << std::endl
                << "Curve instance ID: " << mpPixelData.curveInstanceID << std::endl
                << "Material ID: " << mpPixelData.materialID << std::endl;
        }
        else
        {
            oss << "Background pixel" << std::endl;
        }
        widget.text(oss.str());

        widget.var("Translation", mTranslation);
        widget.var("Scaling", mScaling);
        widget.var("Rotation", mRotation);

        mpPixelDataStaging->unmap();
    }

    widget.dummy("#spacer1", { 1, 20 });
    widget.text("Scene: " + (mpScene ? mpScene->getFilename() : "No scene loaded"));
}
