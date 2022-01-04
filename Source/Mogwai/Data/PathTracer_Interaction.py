from falcor import *

def render_graph_PathTracerGraph():
    g = RenderGraph('PathTracerGraph')
    loadRenderPassLibrary('BSDFViewer.dll')
    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary('DepthPass.dll')
    loadRenderPassLibrary('Antialiasing.dll')
    loadRenderPassLibrary('DebugPasses.dll')
    loadRenderPassLibrary('BlitPass.dll')
    loadRenderPassLibrary('CSM.dll')
    loadRenderPassLibrary('ErrorMeasurePass.dll')
    loadRenderPassLibrary('WhittedRayTracer.dll')
    loadRenderPassLibrary('ExampleBlitPass.dll')
    loadRenderPassLibrary('ForwardLightingPass.dll')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('ImageLoader.dll')
    loadRenderPassLibrary('InteractionPass.dll')
    loadRenderPassLibrary('MegakernelPathTracer.dll')
    loadRenderPassLibrary('MinimalPathTracer.dll')
    loadRenderPassLibrary('PassLibraryTemplate.dll')
    loadRenderPassLibrary('PixelInspectorPass.dll')
    loadRenderPassLibrary('SceneDebugger.dll')
    loadRenderPassLibrary('SkyBox.dll')
    loadRenderPassLibrary('SSAO.dll')
    loadRenderPassLibrary('SVGFPass.dll')
    loadRenderPassLibrary('TemporalDelayPass.dll')
    loadRenderPassLibrary('ToneMapper.dll')
    loadRenderPassLibrary('Utils.dll')
    loadRenderPassLibrary('WireframePass.dll')
    AccumulatePass = createPass('AccumulatePass', {'enableAccumulation': True, 'autoReset': True, 'precisionMode': AccumulatePrecision.Single, 'subFrameCount': 0})
    g.addPass(AccumulatePass, 'AccumulatePass')
    ToneMappingPass = createPass('ToneMapper', {'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': ToneMapOp.Aces, 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': ExposureMode.AperturePriority})
    g.addPass(ToneMappingPass, 'ToneMappingPass')
    GBufferRT = createPass('GBufferRT', {'samplePattern': SamplePattern.Stratified, 'sampleCount': 16, 'disableAlphaTest': False, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': CullMode.CullBack, 'texLOD': LODMode.UseMip0})
    g.addPass(GBufferRT, 'GBufferRT')
    MegakernelPathTracer = createPass('MegakernelPathTracer', {'mSharedParams': PathTracerParams(samplesPerPixel=1, lightSamplesPerVertex=1, maxNonSpecularBounces=3, maxBounces=3, adjustShadingNormals=0, useVBuffer=0, forceAlphaOne=1, useAlphaTest=1, clampSamples=0, useMIS=1, clampThreshold=10.0, useLightsInDielectricVolumes=0, specularRoughnessThreshold=0.25, useBRDFSampling=1, useNestedDielectrics=1, useNEE=1, misHeuristic=1, misPowerExponent=2.0, probabilityAbsorption=0.20000000298023224, useRussianRoulette=0, useFixedSeed=0, useLegacyBSDF=0, disableCaustics=0, rayFootprintMode=0, rayConeMode=2, rayFootprintUseRoughness=0), 'mSelectedSampleGenerator': 1, 'mSelectedEmissiveSampler': EmissiveLightSamplerType.LightBVH, 'mUniformSamplerOptions': EmissiveUniformSamplerOptions(), 'mLightBVHSamplerOptions': LightBVHSamplerOptions(useBoundingCone=True, buildOptions=LightBVHBuilderOptions(splitHeuristicSelection=SplitHeuristic.BinnedSAOH, maxTriangleCountPerLeaf=10, binCount=16, volumeEpsilon=0.0010000000474974513, useLeafCreationCost=True, createLeavesASAP=True, useLightingCones=True, splitAlongLargest=False, useVolumeOverSA=False, allowRefitting=True, usePreintegration=True), useLightingCone=True, disableNodeFlux=False, useUniformTriangleSampling=True, solidAngleBoundMethod=SolidAngleBoundMethod.Sphere)})
    g.addPass(MegakernelPathTracer, 'MegakernelPathTracer')
    InteractionPass = createPass('InteractionPass')
    g.addPass(InteractionPass, 'InteractionPass')
    g.addEdge('GBufferRT.vbuffer', 'MegakernelPathTracer.vbuffer')
    g.addEdge('GBufferRT.posW', 'MegakernelPathTracer.posW')
    g.addEdge('GBufferRT.normW', 'MegakernelPathTracer.normalW')
    g.addEdge('GBufferRT.tangentW', 'MegakernelPathTracer.tangentW')
    g.addEdge('GBufferRT.faceNormalW', 'MegakernelPathTracer.faceNormalW')
    g.addEdge('GBufferRT.viewW', 'MegakernelPathTracer.viewW')
    g.addEdge('GBufferRT.diffuseOpacity', 'MegakernelPathTracer.mtlDiffOpacity')
    g.addEdge('GBufferRT.specRough', 'MegakernelPathTracer.mtlSpecRough')
    g.addEdge('GBufferRT.emissive', 'MegakernelPathTracer.mtlEmissive')
    g.addEdge('GBufferRT.matlExtra', 'MegakernelPathTracer.mtlParams')
    g.addEdge('MegakernelPathTracer.color', 'AccumulatePass.input')
    g.addEdge('AccumulatePass.output', 'ToneMappingPass.src')
    g.addEdge('ToneMappingPass.dst', 'InteractionPass.src')
    g.markOutput('InteractionPass.dst')
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None