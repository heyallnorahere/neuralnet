#pragma once
#include <neuralnet.h>

#if defined(NN_DEBUG) && defined(NN_SUPPORT_vulkan)
#define GUI_ENABLED

#include <GLFW/glfw3.h>
#include <queue>
#include <optional>
#endif

namespace common {
    namespace eval = neuralnet::evaluators;

#ifdef GUI_ENABLED
    struct gui_vtable_t {
        NN_DECLARE_VK_FUNCTION(vkCreateSwapchainKHR);
        NN_DECLARE_VK_FUNCTION(vkDestroySwapchainKHR);
        NN_DECLARE_VK_FUNCTION(vkGetSwapchainImagesKHR);
        NN_DECLARE_VK_FUNCTION(vkAcquireNextImageKHR);
        NN_DECLARE_VK_FUNCTION(vkQueuePresentKHR);

        NN_DECLARE_VK_FUNCTION(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
        NN_DECLARE_VK_FUNCTION(vkGetPhysicalDeviceSurfaceFormatsKHR);
        NN_DECLARE_VK_FUNCTION(vkGetPhysicalDeviceSurfacePresentModesKHR);

        NN_DECLARE_VK_FUNCTION(vkCreateSemaphore);
        NN_DECLARE_VK_FUNCTION(vkDestroySemaphore);

        NN_DECLARE_VK_FUNCTION(vkResetCommandBuffer);
        NN_DECLARE_VK_FUNCTION(vkCmdBlitImage);
    };

    struct command_buffer_t {
        VkCommandBuffer buffer;
        VkFence fence;
        uint32_t queue;
    };

    struct command_pool_t {
        VkCommandPool pool;
        VkQueue queue;

        std::queue<command_buffer_t> buffers;
        std::vector<VkCommandBuffer> allocated;
    };

    struct gui_sync_data_t {
        VkFence fence;
        VkSemaphore image_available, blit_finished;
    };

    struct gui_data_t {
        uint32_t present_index;

        VkSurfaceKHR surface;
        VkSwapchainKHR swapchain;
        VkFormat swapchain_format;
        VkExtent2D image_size;

        std::vector<VkImage> images;
        std::vector<VkFence> image_fences;
        std::unordered_map<uint32_t, command_pool_t> pools;
        std::vector<gui_sync_data_t> sync;
    };

    struct gui_image_context_t {
        VkImage image;
        VkExtent2D size;
        uint32_t z, mip_level, array_layer;

        VkImageLayout layout;
        VkAccessFlags access;
        VkImageAspectFlags aspect_flags;
        VkPipelineStageFlags src_stage, dst_stage;
    };
#endif

    class debug_gui {
    public:
        debug_gui(const std::string& title);
        ~debug_gui();

        debug_gui(const debug_gui&) = delete;
        debug_gui& operator=(const debug_gui&) = delete;

        void update();

#ifdef GUI_ENABLED
        void set_displayed_image(const gui_image_context_t& context);
#endif

        bool is_valid() const { return m_window != nullptr; }

    private:
#ifdef GUI_ENABLED
        static bool set_vulkan_context();
        static void device_chosen(eval::vulkan_context_t* context);
        static void init_finished(eval::vulkan_context_t* context);

        void init_gui(const std::string& title);
        void finish_init_gui();
        void shutdown_gui();

        VkPresentModeKHR choose_present_mode();
        VkSurfaceFormatKHR choose_surface_format();
        std::optional<VkExtent2D> choose_extent(const VkSurfaceCapabilitiesKHR& capabilities);

        void create_window(const std::string& title);
        void fill_vtable();
        void create_sync_objects();

        void invalidate_swapchain();
        std::optional<VkSwapchainKHR> create_swapchain();

        void acquire_image();
        bool present(const command_buffer_t& buffer);

        bool blit(const gui_image_context_t& context);

        command_pool_t& get_pool(uint32_t queue);
        command_buffer_t open_pool(uint32_t queue);
        void close_pool(const command_buffer_t& buffer); // note: this does not submit to a queue

        GLFWwindow* m_window;
        uint32_t m_width, m_height;
        bool m_window_resized;

        eval::vulkan_context_t* m_context;
        gui_vtable_t m_vtable;
        gui_data_t m_data;

        uint32_t m_current_image, m_current_sync_frame;
#endif
    };
}; // namespace common