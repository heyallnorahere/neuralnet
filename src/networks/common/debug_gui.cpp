#include "debug_gui.h"

#ifdef GUI_ENABLED
#include <volk.h>

#include <stack>
#endif

namespace common {
#ifdef GUI_ENABLED
    static constexpr uint32_t s_sync_frames = 2;

    static std::stack<debug_gui*> s_gui_ptrs;

    bool debug_gui::set_vulkan_context() {
        ZoneScoped;

        uint32_t extension_count = 0;
        const char** extensions = glfwGetRequiredInstanceExtensions(&extension_count);

        if (volkInitialize() != VK_SUCCESS) {
            return false;
        }

        auto context = std::make_unique<eval::vulkan_context_t>();
        context->handles.instance_extensions.insert(extensions, &extensions[extension_count]);
        context->handles.device_extensions.insert("VK_KHR_swapchain");

        context->vtable.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        context->vtable.user_callbacks.device_chosen = device_chosen;
        context->vtable.user_callbacks.init_finished = init_finished;

        eval::vulkan_evaluator::set_next_context(std::move(context));
        return true;
    }

    void debug_gui::device_chosen(eval::vulkan_context_t* context) {
        ZoneScoped;
        if (s_gui_ptrs.empty()) {
            std::cout << "debug gui: no pending gui object found" << std::endl;
            return;
        }

        auto& v = context->vtable;
        auto& handles = context->handles;

        auto gui = s_gui_ptrs.top();
        s_gui_ptrs.pop();

        auto _vkGetPhysicalDeviceSurfaceSupportKHR =
            (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)v.vkGetInstanceProcAddr(
                handles.instance, "vkGetPhysicalDeviceSurfaceSupportKHR");

        uint32_t family_count = 0;
        v.vkGetPhysicalDeviceQueueFamilyProperties(handles.physical_device, &family_count, nullptr);

        v.check_result(glfwCreateWindowSurface(handles.instance, gui->m_window, &v.alloc_callbacks,
                                               &gui->m_data.surface));

        std::optional<uint32_t> present_queue;
        for (uint32_t i = 0; i < family_count; i++) {
            VkBool32 supported = VK_FALSE;
            v.check_result(_vkGetPhysicalDeviceSurfaceSupportKHR(handles.physical_device, i,
                                                                 gui->m_data.surface, &supported));

            if (supported == VK_TRUE) {
                present_queue = i;
                break;
            }
        }

        if (present_queue.has_value()) {
            uint32_t index = present_queue.value();
            handles.shared_queue_indices.insert(index);

            gui->m_data.present_index = index;
            gui->m_context = context;

            v.user_callbacks.user_data = gui;
        }
    }

    void debug_gui::init_finished(eval::vulkan_context_t* context) {
        ZoneScoped;

        auto gui = (debug_gui*)context->vtable.user_callbacks.user_data;
        gui->finish_init_gui();
    }
#endif

    debug_gui::debug_gui(const std::string& title) {
        ZoneScoped;

#ifndef GUI_ENABLED
        std::cout << "gui not enabled - not creating ui" << std::endl;
#else
        m_context = nullptr;
        init_gui(title);
#endif
    }

    debug_gui::~debug_gui() {
        ZoneScoped;

#ifdef GUI_ENABLED
        if (m_window != nullptr) {
            shutdown_gui();
        }
#endif
    }

    void debug_gui::update() {
        ZoneScoped;

#ifdef GUI_ENABLED
        glfwPollEvents();
#endif
    }

#ifdef GUI_ENABLED
    void debug_gui::set_displayed_image(const gui_image_context_t& context) {
        ZoneScoped;
        if (context.size.width != m_width || context.size.height != m_height) {
            m_window_resized = true;
            m_width = context.size.width;
            m_height = context.size.height;

            glfwSetWindowSize(m_window, (int)m_width, (int)m_height);
        }

        while (true) {
            if (blit(context)) {
                break;
            }

            // just wait
        }
    }

    void debug_gui::init_gui(const std::string& title) {
        ZoneScoped;
        if (!glfwInit()) {
            std::cerr << "glfw failed to init - skipping gui creation" << std::endl;
            return;
        }

        if (!set_vulkan_context()) {
            std::cerr << "failed to set up vulkan context - skipping gui creation" << std::endl;
            return;
        }

        create_window(title);
        if (m_window == nullptr) {
            std::cerr << "failed to create window - skipping gui creation" << std::endl;
            return;
        }

        s_gui_ptrs.push(this);
    }

    void debug_gui::finish_init_gui() {
        ZoneScoped;

        fill_vtable();
        create_sync_objects();

        m_data.swapchain = VK_NULL_HANDLE;
        create_swapchain();
    }

    void debug_gui::shutdown_gui() {
        ZoneScoped;

        if (m_context != nullptr) {
            const auto& handles = m_context->handles;
            const auto& v = m_context->vtable;

            auto _vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)v.vkGetInstanceProcAddr(
                handles.instance, "vkDestroySurfaceKHR");

            m_vtable.vkDestroySwapchainKHR(handles.device, m_data.swapchain, &v.alloc_callbacks);
            _vkDestroySurfaceKHR(handles.instance, m_data.surface, &v.alloc_callbacks);

            for (const auto& [family, pool] : m_data.pools) {
                v.check_result(v.vkQueueWaitIdle(pool.queue));

                while (!pool.buffers.empty()) {
                    const auto& buffer = pool.buffers.front();
                    v.vkDestroyFence(handles.device, buffer.fence, &v.alloc_callbacks);
                }

                v.vkFreeCommandBuffers(handles.device, pool.pool, (uint32_t)pool.allocated.size(),
                                       pool.allocated.data());

                v.vkDestroyCommandPool(handles.device, pool.pool, &v.alloc_callbacks);
            }

            for (const auto& sync_data : m_data.sync) {
                m_vtable.vkDestroySemaphore(handles.device, sync_data.image_available,
                                            &v.alloc_callbacks);

                m_vtable.vkDestroySemaphore(handles.device, sync_data.blit_finished,
                                            &v.alloc_callbacks);

                v.vkDestroyFence(handles.device, sync_data.fence, &v.alloc_callbacks);
            }
        }

        glfwDestroyWindow(m_window);
    }

    void debug_gui::create_window(const std::string& title) {
        ZoneScoped;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_window_resized = false;
        m_width = 1600;
        m_height = 900;
        m_window = glfwCreateWindow((int)m_width, (int)m_height, title.c_str(), nullptr, nullptr);
    }

#define GET_DEVICE_PROC(name)                                                                      \
    m_vtable.name = (PFN_##name)v.vkGetDeviceProcAddr(handles.device, #name)

#define GET_INSTANCE_PROC(name)                                                                    \
    m_vtable.name = (PFN_##name)v.vkGetInstanceProcAddr(handles.instance, #name)

    void debug_gui::fill_vtable() {
        ZoneScoped;

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        GET_DEVICE_PROC(vkCreateSwapchainKHR);
        GET_DEVICE_PROC(vkDestroySwapchainKHR);
        GET_DEVICE_PROC(vkGetSwapchainImagesKHR);
        GET_DEVICE_PROC(vkAcquireNextImageKHR);
        GET_DEVICE_PROC(vkQueuePresentKHR);

        GET_INSTANCE_PROC(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
        GET_INSTANCE_PROC(vkGetPhysicalDeviceSurfaceFormatsKHR);
        GET_INSTANCE_PROC(vkGetPhysicalDeviceSurfacePresentModesKHR);

        GET_DEVICE_PROC(vkCreateSemaphore);
        GET_DEVICE_PROC(vkDestroySemaphore);

        GET_DEVICE_PROC(vkResetCommandBuffer);
        GET_DEVICE_PROC(vkCmdBlitImage);
    }

    void debug_gui::create_sync_objects() {
        ZoneScoped;

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        m_data.sync.resize(s_sync_frames);
        for (auto& sync : m_data.sync) {
            v.check_result(m_vtable.vkCreateSemaphore(handles.device, &semaphore_info,
                                                      &v.alloc_callbacks, &sync.blit_finished));

            v.check_result(m_vtable.vkCreateSemaphore(handles.device, &semaphore_info,
                                                      &v.alloc_callbacks, &sync.image_available));

            v.check_result(
                v.vkCreateFence(handles.device, &fence_info, &v.alloc_callbacks, &sync.fence));
        }
    }

    VkPresentModeKHR debug_gui::choose_present_mode() {
        ZoneScoped;

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        uint32_t mode_count = 0;
        v.check_result(m_vtable.vkGetPhysicalDeviceSurfacePresentModesKHR(
            handles.physical_device, m_data.surface, &mode_count, nullptr));

        std::vector<VkPresentModeKHR> present_modes(mode_count);
        v.check_result(m_vtable.vkGetPhysicalDeviceSurfacePresentModesKHR(
            handles.physical_device, m_data.surface, &mode_count, present_modes.data()));

        for (auto present_mode : present_modes) {
            if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return present_mode;
            }
        }

        return present_modes[0];
    }

    VkSurfaceFormatKHR debug_gui::choose_surface_format() {
        ZoneScoped;

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        static constexpr VkColorSpaceKHR preferred_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        static const std::vector<VkFormat> preferred_formats = { VK_FORMAT_B8G8R8A8_UNORM,
                                                                 VK_FORMAT_R8G8B8A8_UNORM,
                                                                 VK_FORMAT_B8G8R8_UNORM,
                                                                 VK_FORMAT_R8G8B8_UNORM };

        uint32_t format_count = 0;
        v.check_result(m_vtable.vkGetPhysicalDeviceSurfaceFormatsKHR(
            handles.physical_device, m_data.surface, &format_count, nullptr));

        std::vector<VkSurfaceFormatKHR> surface_formats(format_count);
        v.check_result(m_vtable.vkGetPhysicalDeviceSurfaceFormatsKHR(
            handles.physical_device, m_data.surface, &format_count, surface_formats.data()));

        for (auto preferred_format : preferred_formats) {
            for (const auto& format : surface_formats) {
                if (format.format == preferred_format && format.colorSpace == preferred_space) {
                    return format;
                }
            }
        }

        return surface_formats[0];
    }

    std::optional<VkExtent2D> debug_gui::choose_extent(
        const VkSurfaceCapabilitiesKHR& capabilities) {
        ZoneScoped;

        int width, height;
        glfwGetFramebufferSize(m_window, &width, &height);

        if (width == 0 || height == 0) {
            return {};
        }

        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        VkExtent2D extent;
        extent.width = std::clamp((uint32_t)width, capabilities.minImageExtent.width,
                                  capabilities.maxImageExtent.width);

        extent.height = std::clamp((uint32_t)height, capabilities.maxImageExtent.height,
                                   capabilities.maxImageExtent.height);

        return extent;
    }

    void debug_gui::invalidate_swapchain() {
        ZoneScoped;

        auto old = create_swapchain();
        if (old.has_value()) {
            m_vtable.vkDestroySwapchainKHR(m_context->handles.device, old.value(),
                                           &m_context->vtable.alloc_callbacks);
        }
    }

    std::optional<VkSwapchainKHR> debug_gui::create_swapchain() {
        ZoneScoped;

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        VkSurfaceCapabilitiesKHR capabilities;
        v.check_result(m_vtable.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            handles.physical_device, m_data.surface, &capabilities));

        auto extent = choose_extent(capabilities);
        auto present_mode = choose_present_mode();
        auto format = choose_surface_format();

        if (!extent.has_value()) {
            return {};
        }

        m_data.swapchain_format = format.format;
        m_data.image_size = extent.value();

        uint32_t image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0) {
            image_count = std::min(image_count, capabilities.maxImageCount);
        }

        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = m_data.surface;
        create_info.oldSwapchain = m_data.swapchain;
        create_info.imageFormat = format.format;
        create_info.imageColorSpace = format.colorSpace;
        create_info.presentMode = present_mode;
        create_info.imageExtent = extent.value();
        create_info.minImageCount = image_count;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        uint32_t transfer_queue = m_context->handles.compute_queue_index;
        uint32_t present_queue = m_data.present_index;
        std::vector<uint32_t> indices = { transfer_queue, present_queue };

        if (transfer_queue != present_queue) {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = (uint32_t)indices.size();
            create_info.pQueueFamilyIndices = indices.data();
        } else {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        v.check_result(m_vtable.vkCreateSwapchainKHR(handles.device, &create_info,
                                                     &v.alloc_callbacks, &m_data.swapchain));

        v.check_result(m_vtable.vkGetSwapchainImagesKHR(handles.device, m_data.swapchain,
                                                        &image_count, nullptr));

        m_data.images.resize(image_count);
        m_data.image_fences.resize(image_count, VK_NULL_HANDLE);

        v.check_result(m_vtable.vkGetSwapchainImagesKHR(handles.device, m_data.swapchain,
                                                        &image_count, m_data.images.data()));

        m_current_image = image_count - 1;
        return create_info.oldSwapchain;
    }

    void debug_gui::acquire_image() {
        ZoneScoped;

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        const auto& sync = m_data.sync[m_current_sync_frame];
        while (true) {
            VkResult result = m_vtable.vkAcquireNextImageKHR(
                handles.device, m_data.swapchain, std::numeric_limits<uint64_t>::max(),
                sync.image_available, VK_NULL_HANDLE, &m_current_image);

            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
                invalidate_swapchain();
            } else {
                v.check_result(result);
                break;
            }
        }

        VkFence& image_fence = m_data.image_fences[m_current_image];
        if (image_fence != VK_NULL_HANDLE) {
            v.check_result(v.vkWaitForFences(handles.device, 1, &image_fence, VK_TRUE,
                                             std::numeric_limits<uint64_t>::max()));
            
            image_fence = VK_NULL_HANDLE;
        }
    }

    bool debug_gui::present(const command_buffer_t& buffer) {
        ZoneScoped;
        static const VkPipelineStageFlags wait_flags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

        const auto& transfer_pool = get_pool(m_context->handles.compute_queue_index);
        const auto& present_pool = get_pool(m_data.present_index);
        const auto& sync = m_data.sync[m_current_sync_frame];

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &buffer.buffer;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitDstStageMask = &wait_flags;
        submit_info.pWaitSemaphores = &sync.image_available;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &sync.blit_finished;

        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &m_data.swapchain;
        present_info.pImageIndices = &m_current_image;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &sync.blit_finished;

        v.check_result(v.vkResetFences(handles.device, 1, &sync.fence));
        v.check_result(v.vkQueueSubmit(transfer_pool.queue, 1, &submit_info, sync.fence));
        VkResult result = m_vtable.vkQueuePresentKHR(present_pool.queue, &present_info);

        m_current_sync_frame++;
        m_current_sync_frame %= s_sync_frames;
        m_data.image_fences[m_current_image] = sync.fence;

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_window_resized) {
            invalidate_swapchain();

            m_window_resized = false;
            return false;
        }

        v.check_result(result);
        return true;
    }

    bool debug_gui::blit(const gui_image_context_t& context) {
        ZoneScoped;
        const auto& v = m_context->vtable;

        acquire_image();
        VkImage current_image = m_data.images[m_current_image];

        static constexpr VkImageLayout src_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        static constexpr VkAccessFlags src_access = VK_ACCESS_TRANSFER_READ_BIT;
        static constexpr VkImageLayout dst_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        static constexpr VkAccessFlags dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
        static constexpr VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        VkImageMemoryBarrier src_src{};
        src_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        src_src.image = context.image;
        src_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        src_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        src_src.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        src_src.newLayout = src_layout;
        src_src.srcAccessMask = context.access;
        src_src.dstAccessMask = src_access;
        src_src.subresourceRange.aspectMask = context.aspect_flags;
        src_src.subresourceRange.baseArrayLayer = context.array_layer;
        src_src.subresourceRange.layerCount = 1;
        src_src.subresourceRange.baseMipLevel = context.mip_level;
        src_src.subresourceRange.levelCount = 1;

        VkImageMemoryBarrier src_dst{};
        src_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        src_dst.image = context.image;
        src_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        src_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        src_dst.oldLayout = src_layout;
        src_dst.newLayout = context.layout;
        src_dst.srcAccessMask = src_access;
        src_dst.dstAccessMask = context.access;
        src_dst.subresourceRange.aspectMask = context.aspect_flags;
        src_dst.subresourceRange.baseArrayLayer = context.array_layer;
        src_dst.subresourceRange.layerCount = 1;
        src_dst.subresourceRange.baseMipLevel = context.mip_level;
        src_dst.subresourceRange.levelCount = 1;

        VkImageMemoryBarrier dst_src{};
        dst_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        dst_src.image = current_image;
        dst_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dst_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dst_src.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        dst_src.newLayout = dst_layout;
        dst_src.srcAccessMask = 0;
        dst_src.dstAccessMask = dst_access;
        dst_src.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        dst_src.subresourceRange.baseArrayLayer = 0;
        dst_src.subresourceRange.layerCount = 1;
        dst_src.subresourceRange.baseMipLevel = 0;
        dst_src.subresourceRange.levelCount = 1;

        VkImageMemoryBarrier dst_dst{};
        dst_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        dst_dst.image = current_image;
        dst_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dst_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dst_dst.oldLayout = dst_layout;
        dst_dst.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        dst_dst.srcAccessMask = 0;
        dst_dst.dstAccessMask = 0;
        dst_dst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        dst_dst.subresourceRange.baseArrayLayer = 0;
        dst_dst.subresourceRange.layerCount = 1;
        dst_dst.subresourceRange.baseMipLevel = 0;
        dst_dst.subresourceRange.levelCount = 1;

        VkImageBlit blit{};
        blit.srcOffsets[0].z = (int32_t)context.z;
        blit.srcOffsets[1].x = (int32_t)context.size.width;
        blit.srcOffsets[1].y = (int32_t)context.size.height;
        blit.srcOffsets[1].z = 1;
        blit.srcSubresource.aspectMask = context.aspect_flags;
        blit.srcSubresource.baseArrayLayer = context.array_layer;
        blit.srcSubresource.layerCount = 1;
        blit.srcSubresource.mipLevel = context.mip_level;
        blit.dstOffsets[1].x = (int32_t)m_data.image_size.width;
        blit.dstOffsets[1].y = (int32_t)m_data.image_size.height;
        blit.dstOffsets[1].z = 1;
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;
        blit.dstSubresource.mipLevel = 0;

        std::vector<VkImageMemoryBarrier> src = { src_src, dst_src };
        std::vector<VkImageMemoryBarrier> dst = { src_dst, dst_dst };

        auto buffer = open_pool(m_context->handles.compute_queue_index);
        v.vkCmdPipelineBarrier(buffer.buffer, context.src_stage, stage, 0, 0, nullptr, 0, nullptr,
                               (uint32_t)src.size(), src.data());

        m_vtable.vkCmdBlitImage(buffer.buffer, context.image, src_layout, current_image, dst_layout,
                                1, &blit, VK_FILTER_NEAREST);

        v.vkCmdPipelineBarrier(buffer.buffer, stage, context.dst_stage, 0, 0, nullptr, 0, nullptr,
                               (uint32_t)dst.size(), dst.data());

        close_pool(buffer);
        return present(buffer);
    }

    command_pool_t& debug_gui::get_pool(uint32_t queue) {
        ZoneScoped;

        bool exists = m_data.pools.contains(queue);
        auto& pool = m_data.pools[queue];

        if (!exists) {
            const auto& v = m_context->vtable;
            const auto& handles = m_context->handles;

            VkCommandPoolCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            create_info.queueFamilyIndex = queue;
            create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

            v.vkGetDeviceQueue(handles.device, queue, 0, &pool.queue);
            v.check_result(v.vkCreateCommandPool(handles.device, &create_info, &v.alloc_callbacks,
                                                 &pool.pool));
        }

        return pool;
    }

    command_buffer_t debug_gui::open_pool(uint32_t queue) {
        ZoneScoped;
        auto& pool = get_pool(queue);

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        if (!pool.buffers.empty()) {
            const auto& front = pool.buffers.front();

            VkResult status = v.vkGetFenceStatus(handles.device, front.fence);
            if (status == VK_SUCCESS) {
                v.check_result(v.vkResetFences(handles.device, 1, &front.fence));
                v.check_result(m_vtable.vkResetCommandBuffer(front.buffer, 0));

                auto result = front;
                pool.buffers.pop();
                return result;
            }
        }

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = pool.pool;
        alloc_info.commandBufferCount = 1;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        command_buffer_t buffer;
        buffer.queue = queue;

        v.check_result(v.vkAllocateCommandBuffers(handles.device, &alloc_info, &buffer.buffer));
        v.check_result(v.vkBeginCommandBuffer(buffer.buffer, &begin_info));

        v.check_result(
            v.vkCreateFence(handles.device, &fence_info, &v.alloc_callbacks, &buffer.fence));

        return buffer;
    }

    void debug_gui::close_pool(const command_buffer_t& buffer) {
        ZoneScoped;

        const auto& v = m_context->vtable;
        v.check_result(v.vkEndCommandBuffer(buffer.buffer));

        auto& pool = get_pool(buffer.queue);
        pool.buffers.push(buffer);
    }
#endif
} // namespace common
