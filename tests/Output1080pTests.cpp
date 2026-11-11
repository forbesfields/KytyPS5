#include "common/emulatorConfig.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "Output1080pTests: %s\n", message);
		std::exit(1);
	}
}

struct DisplayCase {
	uint32_t width;
	uint32_t height;
	uint32_t attribute3;
	uint32_t resolution;
};

constexpr std::array cases {
    DisplayCase {1280, 720, 0, 2},
    DisplayCase {1920, 1080, 0, 2},
    DisplayCase {1920, 1080, 4, 1},
    DisplayCase {3839, 2159, 4, 1},
    DisplayCase {3840, 1080, 4, 2},
    DisplayCase {1920, 2160, 4, 2},
    DisplayCase {3840, 2160, 0, 2},
    DisplayCase {3840, 2160, 4, 2},
    DisplayCase {7680, 4320, 4, 2},
    // Astro Bot's logged ATTRIBUTE3: detection allowed, Trinity unsupported.
    DisplayCase {1920, 1080, 0x00040014, 1},
    DisplayCase {3840, 2160, 0x00040014, 2},
};

void CheckResolutions(bool forced) {
	for (const auto& test: cases) {
		Check(Config::GetVideoOutResolution(test.width, test.height, test.attribute3) ==
		          (forced ? 1u : test.resolution),
		      "display query must honor the option, permission and 4K boundaries");
	}
}

} // namespace

int main(int argc, char* argv[]) {
	const bool killed = argc == 2 && std::strcmp(argv[1], "--killed") == 0;
	Config::Initialize();
	Config::ConfigOptions options;
	Check(!options.output_1080p_enabled, "1080p output must default off");
	Config::Load(options);
	Check(!Config::Output1080pEnabled(), "default config must keep legacy output selection");
	CheckResolutions(false);

	options.output_1080p_enabled = true;
	options.screen_width = 3840;
	options.screen_height = 2160;
	options.hdr_enabled = true;
	options.pro_enabled = true;
	options.vblank_frequency = 120;
	Config::Load(options);
	Check(Config::Output1080pEnabled() == !killed, "kill switch must override the option");
	CheckResolutions(!killed);
	Check(Config::GetScreenWidth() == 3840 && Config::GetScreenHeight() == 2160,
	      "guest output must not resize the host window");
	Check(Config::HdrEnabled() && Config::Ps5ProEnabled() && Config::GetVblankFrequency() == 120,
	      "display resolution must not change HDR, console mode or refresh rate");

	// Changes after startup must not change the cached switch.
#if defined(_WIN32)
	_putenv_s("KYTY_PERF_OUTPUT_1080P", killed ? "1" : "0");
#else
	setenv("KYTY_PERF_OUTPUT_1080P", killed ? "1" : "0", 1);
#endif
	Config::Load(options);
	CheckResolutions(!killed);
	options.output_1080p_enabled = false;
	Config::Load(options);
	CheckResolutions(false);
	Config::Shutdown();
	std::printf("Output1080pTests: passed (option off/on, kill switch %s, 11 display cases)\n",
	            killed ? "off" : "on");
	return 0;
}
