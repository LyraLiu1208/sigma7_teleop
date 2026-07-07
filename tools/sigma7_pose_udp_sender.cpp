#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "dhdc.h"
#include "drdc.h"

namespace {

volatile sig_atomic_t g_running = 1;

void handle_signal(int) {
  g_running = 0;
}

struct Options {
  std::string host = "127.0.0.1";
  int port = 5005;
  double hz = 200.0;
  int stdout_every = 100;
  double max_seconds = 0.0;
  bool skip_init = false;
};

bool parse_int(const char* text, int* out) {
  if (text == nullptr || out == nullptr) {
    return false;
  }
  char* end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

bool parse_double(const char* text, double* out) {
  if (text == nullptr || out == nullptr) {
    return false;
  }
  char* end = nullptr;
  double value = std::strtod(text, &end);
  if (end == text || *end != '\0') {
    return false;
  }
  *out = value;
  return true;
}

void print_usage(const char* program) {
  std::fprintf(
      stderr,
      "Usage: %s [--host HOST] [--port PORT] [--hz RATE] [--stdout-every N] [--max-seconds S]\n",
      program);
}

bool parse_args(int argc, char* argv[], Options* options) {
  if (options == nullptr) {
    return false;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--host" && i + 1 < argc) {
      options->host = argv[++i];
      continue;
    }
    if (arg == "--port" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options->port)) {
        return false;
      }
      continue;
    }
    if (arg == "--hz" && i + 1 < argc) {
      if (!parse_double(argv[++i], &options->hz)) {
        return false;
      }
      continue;
    }
    if (arg == "--stdout-every" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options->stdout_every)) {
        return false;
      }
      continue;
    }
    if (arg == "--max-seconds" && i + 1 < argc) {
      if (!parse_double(argv[++i], &options->max_seconds)) {
        return false;
      }
      continue;
    }
    if (arg == "--skip-init") {
      options->skip_init = true;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    return false;
  }
  return options->port > 0 && options->hz > 0.0;
}

std::string build_packet(
    unsigned long long sequence,
    double packet_timestamp,
    const double position[3],
    const double rotation[3][3],
    double gripper_angle_rad,
    double linear_velocity[3],
    double angular_velocity_rad[3],
    double gripper_linear_velocity,
    unsigned int buttons) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(9);
  stream << "{";
  stream << "\"schema_version\":\"sigma7_pose_udp_json_v1\",";
  stream << "\"source\":\"sigma7_sdk_direct\",";
  stream << "\"sequence\":" << sequence << ",";
  stream << "\"packet_timestamp\":" << packet_timestamp << ",";
  stream << "\"valid\":true,";
  stream << "\"position\":["
         << position[0] << "," << position[1] << "," << position[2] << "],";
  stream << "\"orientation_frame\":[";
  for (int row = 0; row < 3; ++row) {
    if (row > 0) {
      stream << ",";
    }
    stream << "[";
    for (int col = 0; col < 3; ++col) {
      if (col > 0) {
        stream << ",";
      }
      stream << rotation[row][col];
    }
    stream << "]";
  }
  stream << "],";
  stream << "\"gripper_angle_rad\":" << gripper_angle_rad << ",";
  stream << "\"linear_velocity\":["
         << linear_velocity[0] << "," << linear_velocity[1] << "," << linear_velocity[2] << "],";
  stream << "\"angular_velocity_rad\":["
         << angular_velocity_rad[0] << "," << angular_velocity_rad[1] << "," << angular_velocity_rad[2] << "],";
  stream << "\"gripper_linear_velocity\":" << gripper_linear_velocity << ",";
  stream << "\"buttons\":" << buttons;
  stream << "}";
  return stream.str();
}

}  // namespace

int main(int argc, char* argv[]) {
  Options options;
  if (!parse_args(argc, argv, &options)) {
    print_usage(argv[0]);
    return 1;
  }

  signal(SIGINT, handle_signal);
  signal(SIGTERM, handle_signal);

  const int socket_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd < 0) {
    std::perror("socket");
    return 1;
  }

  sockaddr_in destination;
  std::memset(&destination, 0, sizeof(destination));
  destination.sin_family = AF_INET;
  destination.sin_port = htons(static_cast<uint16_t>(options.port));
  if (::inet_pton(AF_INET, options.host.c_str(), &destination.sin_addr) != 1) {
    std::fprintf(stderr, "error: invalid host %s\n", options.host.c_str());
    ::close(socket_fd);
    return 1;
  }

  dhdEnableExpertMode();

  if (drdOpen() < 0) {
    std::fprintf(stderr, "error: %s\n", dhdErrorGetLastStr());
    ::close(socket_fd);
    return 1;
  }

  if (!drdIsSupported()) {
    std::fprintf(stderr, "error: unsupported device\n");
    drdClose();
    ::close(socket_fd);
    return 1;
  }

  bool regulation_thread_running = false;
  if (!options.skip_init) {
    const double null_pose[DHD_MAX_DOF] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (!drdIsInitialized()) {
      std::printf("initializing device...\n");
      std::fflush(stdout);
      if (drdAutoInit() < 0) {
        std::fprintf(stderr, "error: auto-initialization failed (%s)\n", dhdErrorGetLastStr());
        drdClose();
        ::close(socket_fd);
        return 1;
      }
    }
    if (drdStart() < 0) {
      std::fprintf(stderr, "error: failed to start regulation thread (%s)\n", dhdErrorGetLastStr());
      drdClose();
      ::close(socket_fd);
      return 1;
    }
    regulation_thread_running = true;
    std::printf("moving device to centered workspace...\n");
    std::fflush(stdout);
    if (drdMoveTo(const_cast<double*>(null_pose)) < 0) {
      std::fprintf(stderr, "error: failed to move to center (%s)\n", dhdErrorGetLastStr());
      drdStop();
      drdClose();
      ::close(socket_fd);
      return 1;
    }

    if (drdSetForceAndTorqueAndGripperForce(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) < 0) {
      std::fprintf(stderr, "error: cannot request null force (%s)\n", dhdErrorGetLastStr());
      drdStop();
      drdClose();
      ::close(socket_fd);
      return 1;
    }

    if (drdRegulatePos(false) < 0 || drdRegulateRot(false) < 0 || drdRegulateGrip(false) < 0) {
      std::fprintf(stderr, "error: cannot release device regulation (%s)\n", dhdErrorGetLastStr());
      drdStop();
      drdClose();
      ::close(socket_fd);
      return 1;
    }
  }

  std::printf("%s device detected\n", dhdGetSystemName());
  std::printf(
      "streaming to udp://%s:%d at %.1f Hz, press Ctrl-C to stop\n",
      options.host.c_str(),
      options.port,
      options.hz);
  std::fflush(stdout);

  const double period_seconds = 1.0 / options.hz;
  const auto start_time = std::chrono::steady_clock::now();
  auto next_tick = start_time;
  unsigned long long sequence = 0;

  while (g_running) {
    if (regulation_thread_running) {
      drdWaitForTick();
    }

    const double packet_timestamp = dhdGetTime();

    if (regulation_thread_running) {
      if (drdSetForceAndTorqueAndGripperForce(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) < 0) {
        std::fprintf(stderr, "warning: cannot set zero force (%s)\n", dhdErrorGetLastStr());
        break;
      }
    } else {
      if (dhdSetForceAndTorqueAndGripperForce(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) < 0) {
        std::fprintf(stderr, "warning: cannot set zero force (%s)\n", dhdErrorGetLastStr());
        break;
      }
    }

    double position[3] = {0.0, 0.0, 0.0};
    double rotation[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    double linear_velocity[3] = {0.0, 0.0, 0.0};
    double angular_velocity_rad[3] = {0.0, 0.0, 0.0};
    double gripper_angle_rad = 0.0;
    double gripper_linear_velocity = 0.0;

    if (dhdGetPosition(&position[0], &position[1], &position[2]) < 0) {
      std::fprintf(stderr, "warning: dhdGetPosition failed: %s\n", dhdErrorGetLastStr());
      break;
    }
    if (dhdGetOrientationFrame(rotation) < 0) {
      std::fprintf(stderr, "warning: dhdGetOrientationFrame failed: %s\n", dhdErrorGetLastStr());
      break;
    }
    if (dhdGetGripperAngleRad(&gripper_angle_rad) < 0) {
      gripper_angle_rad = 0.0;
    }
    if (dhdGetLinearVelocity(&linear_velocity[0], &linear_velocity[1], &linear_velocity[2]) < 0) {
      linear_velocity[0] = 0.0;
      linear_velocity[1] = 0.0;
      linear_velocity[2] = 0.0;
    }
    if (dhdGetAngularVelocityRad(&angular_velocity_rad[0], &angular_velocity_rad[1], &angular_velocity_rad[2]) < 0) {
      angular_velocity_rad[0] = 0.0;
      angular_velocity_rad[1] = 0.0;
      angular_velocity_rad[2] = 0.0;
    }
    if (dhdGetGripperLinearVelocity(&gripper_linear_velocity) < 0) {
      gripper_linear_velocity = 0.0;
    }
    const unsigned int buttons = dhdGetButtonMask();

    const std::string payload = build_packet(
        sequence,
        packet_timestamp,
        position,
        rotation,
        gripper_angle_rad,
        linear_velocity,
        angular_velocity_rad,
        gripper_linear_velocity,
        buttons);

    const ssize_t sent = ::sendto(
        socket_fd,
        payload.data(),
        payload.size(),
        0,
        reinterpret_cast<const sockaddr*>(&destination),
        sizeof(destination));
    if (sent < 0) {
      std::perror("sendto");
      break;
    }

    if (options.stdout_every > 0 && (sequence % static_cast<unsigned long long>(options.stdout_every) == 0ULL)) {
      std::printf(
          "seq=%llu pos=(%+.4f %+.4f %+.4f) grip=%+.4f linvel=(%+.4f %+.4f %+.4f)\n",
          sequence,
          position[0],
          position[1],
          position[2],
          gripper_angle_rad,
          linear_velocity[0],
          linear_velocity[1],
          linear_velocity[2]);
      std::fflush(stdout);
    }

    ++sequence;
    if (options.max_seconds > 0.0) {
      const auto now = std::chrono::steady_clock::now();
      const std::chrono::duration<double> elapsed = now - start_time;
      if (elapsed.count() >= options.max_seconds) {
        break;
      }
    }

    if (!regulation_thread_running) {
      next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(period_seconds));
      std::this_thread::sleep_until(next_tick);
    }
  }

  if (regulation_thread_running) {
    drdSetForceAndTorqueAndGripperForce(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    drdStop();
  } else {
    dhdSetForceAndTorqueAndGripperForce(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }
  drdClose();
  ::close(socket_fd);
  std::printf("sender stopped\n");
  return 0;
}
