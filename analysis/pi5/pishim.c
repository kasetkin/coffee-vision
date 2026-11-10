/* Pi 5 benchmark only: a Vulkan layer that lets Dawn accept V3DV. It reports the four
 * shader*ArrayDynamicIndexing features (which WGSL cannot use: it has no descriptor arrays) as present,
 * and removes them again from vkCreateDevice so the driver never sees them. With PISHIM_LIMITS=1 it also
 * reports WebGPU core's texture limits (8192) and 8 fragment outputs. */
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define EXPORT __attribute__((visibility("default")))

static PFN_vkGetInstanceProcAddr next_gipa;
static PFN_vkGetDeviceProcAddr next_gdpa;
static PFN_vkCreateShaderModule next_csm;
static int n_modules;
static VkInstance g_inst;
static PFN_vkGetPhysicalDeviceFeatures next_feat;
static PFN_vkGetPhysicalDeviceFeatures2 next_feat2;
static PFN_vkGetPhysicalDeviceProperties next_props;
static PFN_vkGetPhysicalDeviceProperties2 next_props2;

static void set_dyn(VkPhysicalDeviceFeatures *f, VkBool32 v) {
    f->shaderUniformBufferArrayDynamicIndexing = v;
    f->shaderSampledImageArrayDynamicIndexing = v;
    f->shaderStorageBufferArrayDynamicIndexing = v;
    f->shaderStorageImageArrayDynamicIndexing = v;
}
static void fix_limits(VkPhysicalDeviceLimits *l) {
    const char *e = getenv("PISHIM_LIMITS");
    if (!e || strcmp(e, "1")) return;
    if (l->maxImageDimension1D < 8192) l->maxImageDimension1D = 8192;
    if (l->maxImageDimension2D < 8192) l->maxImageDimension2D = 8192;
    if (l->maxFragmentOutputAttachments < 8) l->maxFragmentOutputAttachments = 8;
    if (l->maxImageDimensionCube < 8192) l->maxImageDimensionCube = 8192;
    if (l->maxViewportDimensions[0] < 8192) l->maxViewportDimensions[0] = 8192;
    if (l->maxViewportDimensions[1] < 8192) l->maxViewportDimensions[1] = 8192;
    if (l->maxFramebufferWidth < 8192) l->maxFramebufferWidth = 8192;
    if (l->maxFramebufferHeight < 8192) l->maxFramebufferHeight = 8192;
    if (l->viewportBoundsRange[1] < 16383) { l->viewportBoundsRange[0] = -16384; l->viewportBoundsRange[1] = 16383; }
    if (l->maxVertexOutputComponents < 128) l->maxVertexOutputComponents = 128;
    if (l->maxFragmentInputComponents < 128) l->maxFragmentInputComponents = 128;
}

static void VKAPI_CALL l_GetPhysicalDeviceFeatures(VkPhysicalDevice pd, VkPhysicalDeviceFeatures *f) {
    next_feat(pd, f); set_dyn(f, VK_TRUE);
}
static void VKAPI_CALL l_GetPhysicalDeviceFeatures2(VkPhysicalDevice pd, VkPhysicalDeviceFeatures2 *f) {
    next_feat2(pd, f); set_dyn(&f->features, VK_TRUE);
}
static void VKAPI_CALL l_GetPhysicalDeviceProperties(VkPhysicalDevice pd, VkPhysicalDeviceProperties *p) {
    next_props(pd, p); fix_limits(&p->limits);
}
static void VKAPI_CALL l_GetPhysicalDeviceProperties2(VkPhysicalDevice pd, VkPhysicalDeviceProperties2 *p) {
    next_props2(pd, p); fix_limits(&p->properties.limits);
}

static VkResult VKAPI_CALL l_CreateInstance(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *a,
                                            VkInstance *inst) {
    VkLayerInstanceCreateInfo *li = (VkLayerInstanceCreateInfo *)ci->pNext;
    while (li && !(li->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && li->function == VK_LAYER_LINK_INFO))
        li = (VkLayerInstanceCreateInfo *)li->pNext;
    if (!li) return VK_ERROR_INITIALIZATION_FAILED;
    next_gipa = li->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    li->u.pLayerInfo = li->u.pLayerInfo->pNext;
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)next_gipa(NULL, "vkCreateInstance");
    VkResult r = create(ci, a, inst);
    if (r != VK_SUCCESS) return r;
    g_inst = *inst;
    next_feat = (PFN_vkGetPhysicalDeviceFeatures)next_gipa(*inst, "vkGetPhysicalDeviceFeatures");
    next_feat2 = (PFN_vkGetPhysicalDeviceFeatures2)next_gipa(*inst, "vkGetPhysicalDeviceFeatures2");
    if (!next_feat2) next_feat2 = (PFN_vkGetPhysicalDeviceFeatures2)next_gipa(*inst, "vkGetPhysicalDeviceFeatures2KHR");
    next_props = (PFN_vkGetPhysicalDeviceProperties)next_gipa(*inst, "vkGetPhysicalDeviceProperties");
    next_props2 = (PFN_vkGetPhysicalDeviceProperties2)next_gipa(*inst, "vkGetPhysicalDeviceProperties2");
    if (!next_props2) next_props2 = (PFN_vkGetPhysicalDeviceProperties2)next_gipa(*inst, "vkGetPhysicalDeviceProperties2KHR");
    fprintf(stderr, "pishim: active (limits %s)\n", getenv("PISHIM_LIMITS") ? getenv("PISHIM_LIMITS") : "0");
    return r;
}

static VkResult VKAPI_CALL l_CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci,
                                          const VkAllocationCallbacks *a, VkDevice *dev) {
    VkLayerDeviceCreateInfo *li = (VkLayerDeviceCreateInfo *)ci->pNext;
    while (li && !(li->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && li->function == VK_LAYER_LINK_INFO))
        li = (VkLayerDeviceCreateInfo *)li->pNext;
    if (!li) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = li->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    next_gdpa = li->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    li->u.pLayerInfo = li->u.pLayerInfo->pNext;
    PFN_vkCreateDevice create = (PFN_vkCreateDevice)gipa(g_inst, "vkCreateDevice");
    if (ci->pEnabledFeatures) set_dyn((VkPhysicalDeviceFeatures *)ci->pEnabledFeatures, VK_FALSE);
    for (VkBaseOutStructure *s = (VkBaseOutStructure *)ci->pNext; s; s = s->pNext)
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
            set_dyn(&((VkPhysicalDeviceFeatures2 *)s)->features, VK_FALSE);
    VkResult r = create(pd, ci, a, dev);
    fprintf(stderr, "pishim: vkCreateDevice -> %d\n", r);
    return r;
}

/* Each SPIR-V module Dawn creates: its index and size to stderr, and with PISHIM_DUMP=<dir> its words to
 * <dir>/<index>.spv, so a shader the driver is slow to compile can be found and read. */
static VkResult VKAPI_CALL l_CreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo *ci,
                                                const VkAllocationCallbacks *a, VkShaderModule *m) {
    int i = ++n_modules;
    fprintf(stderr, "pishim: shader module %d, %zu bytes of SPIR-V\n", i, ci->codeSize);
    const char *dir = getenv("PISHIM_DUMP");
    if (dir) {
        char path[512];
        snprintf(path, sizeof path, "%s/%04d.spv", dir, i);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(ci->pCode, 1, ci->codeSize, f); fclose(f); }
    }
    return next_csm(d, ci, a, m);
}
static PFN_vkVoidFunction VKAPI_CALL l_GetDeviceProcAddr(VkDevice d, const char *n) {
    if (!strcmp(n, "vkCreateShaderModule")) {
        if (!next_csm) next_csm = (PFN_vkCreateShaderModule)next_gdpa(d, n);
        return (PFN_vkVoidFunction)l_CreateShaderModule;
    }
    return next_gdpa ? next_gdpa(d, n) : NULL;
}
static PFN_vkVoidFunction VKAPI_CALL l_GetInstanceProcAddr(VkInstance i, const char *n) {
#define H(name, fn) if (!strcmp(n, name)) return (PFN_vkVoidFunction)fn;
    H("vkGetInstanceProcAddr", l_GetInstanceProcAddr)
    H("vkGetDeviceProcAddr", l_GetDeviceProcAddr)
    H("vkCreateInstance", l_CreateInstance)
    H("vkCreateDevice", l_CreateDevice)
    H("vkGetPhysicalDeviceFeatures", l_GetPhysicalDeviceFeatures)
    H("vkGetPhysicalDeviceFeatures2", l_GetPhysicalDeviceFeatures2)
    H("vkGetPhysicalDeviceFeatures2KHR", l_GetPhysicalDeviceFeatures2)
    H("vkGetPhysicalDeviceProperties", l_GetPhysicalDeviceProperties)
    H("vkGetPhysicalDeviceProperties2", l_GetPhysicalDeviceProperties2)
    H("vkGetPhysicalDeviceProperties2KHR", l_GetPhysicalDeviceProperties2)
    return next_gipa ? next_gipa(i, n) : NULL;
}

EXPORT VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *p) {
    if (p->loaderLayerInterfaceVersion > 2) p->loaderLayerInterfaceVersion = 2;
    p->pfnGetInstanceProcAddr = l_GetInstanceProcAddr;
    p->pfnGetDeviceProcAddr = l_GetDeviceProcAddr;
    p->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}
