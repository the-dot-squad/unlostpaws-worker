#!/usr/bin/env bash
# UnLostPaws vision worker — setup and operator entry point.
#
# Usage:
#   ./tools/run.sh [subcommand] [options]
#   (Runs interactive Docker Compose setup if no subcommand is provided)
#
# Subcommands:
#   doctor       - Run Python hardware preflight checks and print recommendations
#   smoke        - Run smoke tests to verify model loading and execution
#   benchmark    - Run benchmark tests on model performance
#   export       - Export models to ONNX
#   validate     - Validate inputs or dataset records
#   setup        - Run the interactive Docker Compose setup wizard

set -euo pipefail

# ANSI color codes for rich styling
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0;37m' # Normal Color
NC_BOLD='\033[1m'
NC_DIM='\033[2m'
RESET='\033[0m'

GITHUB_RAW_URL="https://raw.githubusercontent.com/the-dot-squad/unlostpaws-worker/main"

# Get current OS and architecture
get_os() { uname -s; }
get_arch() { uname -m; }

# Helper to read input from terminal (even if script is piped through curl)
prompt_user() {
  local prompt_msg="$1"
  local default_val="$2"
  local user_input=""
  
  if [ -c /dev/tty ] && { true; } 2>/dev/null < /dev/tty; then
    if [[ -n "$default_val" ]]; then
      read -rp "$(printf "${NC_BOLD}%s${RESET} [${default_val}]: " "$prompt_msg")" user_input < /dev/tty
    else
      read -rp "$(printf "${NC_BOLD}%s${RESET}: " "$prompt_msg")" user_input < /dev/tty
    fi
  elif [ -t 0 ]; then
    if [[ -n "$default_val" ]]; then
      read -rp "$(printf "${NC_BOLD}%s${RESET} [${default_val}]: " "$prompt_msg")" user_input
    else
      read -rp "$(printf "${NC_BOLD}%s${RESET}: " "$prompt_msg")" user_input
    fi
  else
    user_input="$default_val"
  fi
  
  if [[ -z "$user_input" ]]; then
    user_input="$default_val"
  fi
  echo "$user_input"
}

# Helper to get yes/no confirmation
confirm() {
  local prompt_msg="$1"
  local default_val="${2:-N}"
  local user_input
  user_input=$(prompt_user "$prompt_msg" "$default_val")
  if [[ "${user_input:0:1}" == "y" || "${user_input:0:1}" == "Y" ]]; then
    return 0
  else
    return 1
  fi
}

# Check if Docker is installed
check_docker() {
  if command -v docker >/dev/null 2>&1; then
    docker --version | awk '{print $3}' | tr -d ','
  else
    echo ""
  fi
}

# Check if Compose is available
check_compose() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
  else
    echo ""
  fi
}

# Detect CPU cores
get_cpu_cores() {
  local os
  os=$(get_os)
  if [[ "$os" == "Darwin" ]]; then
    sysctl -n hw.ncpu 2>/dev/null || echo "Unknown"
  else
    nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "Unknown"
  fi
}

# Detect total system RAM in GB
get_total_ram_gb() {
  local os
  os=$(get_os)
  if [[ "$os" == "Darwin" ]]; then
    local ram_bytes
    ram_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    echo $((ram_bytes / 1024 / 1024 / 1024))
  else
    local ram_kb
    ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}' 2>/dev/null || echo 0)
    echo $((ram_kb / 1024 / 1024))
  fi
}

# Detect free disk space in GB
get_free_disk_gb() {
  local disk_kb
  disk_kb=$(df -k . 2>/dev/null | tail -n 1 | awk '{print $4}')
  if [[ -n "$disk_kb" ]]; then
    echo $((disk_kb / 1024 / 1024))
  else
    echo "Unknown"
  fi
}

# Detect NVIDIA GPU details
HAS_NVIDIA=false
GPU_NAME=""
GPU_VRAM=""
detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
      HAS_NVIDIA=true
      GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
      local vram_mb
      vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d '[[:space:]]')
      GPU_VRAM=$((vram_mb / 1024))
    fi
  fi
}

# Download template files from GitHub
download_file() {
  local url="$1"
  local dest="$2"
  
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$dest" "$url"
  else
    echo -e "${RED}Error: Neither curl nor wget is installed. Please install one to proceed.${RESET}" >&2
    exit 1
  fi
}

# Update a configuration option in .env file
update_env_var() {
  local key="$1"
  local value="$2"
  local file="$3"
  local os
  os=$(get_os)
  
  if grep -E "^[# ]*${key}=" "$file" >/dev/null 2>&1; then
    if [[ "$os" == "Darwin" ]]; then
      sed -i "" -E "s|^[# ]*${key}=.*|${key}=${value}|" "$file"
    else
      sed -i -E "s|^[# ]*${key}=.*|${key}=${value}|" "$file"
    fi
  else
    echo "${key}=${value}" >> "$file"
  fi
}

# Interactive setup wizard
run_setup() {
  local os
  local arch
  local cores
  local ram_gb
  local disk_gb
  local docker_ver
  local compose_cmd

  os=$(get_os)
  arch=$(get_arch)
  cores=$(get_cpu_cores)
  ram_gb=$(get_total_ram_gb)
  disk_gb=$(get_free_disk_gb)
  docker_ver=$(check_docker)
  compose_cmd=$(check_compose)
  detect_gpu

  echo -e "\n${BLUE}========================================================${RESET}"
  echo -e "${BLUE}        UnLostPaws Vision Worker Setup Wizard           ${RESET}"
  echo -e "${BLUE}========================================================${RESET}"
  
  # Print System Information
  echo -e "\n${NC_BOLD}Detected System Specifications:${RESET}"
  echo -e "  - OS / Arch:  $os / $arch"
  echo -e "  - CPU Cores:  $cores"
  echo -e "  - System RAM: ${ram_gb} GB"
  echo -e "  - Free Disk:  ${disk_gb} GB"
  if $HAS_NVIDIA; then
    echo -e "  - GPU:        $GPU_NAME (${GPU_VRAM} GB VRAM)"
  else
    echo -e "  - GPU:        None / No NVIDIA driver"
  fi

  # Requirements Validation Checklist
  echo -e "\n${NC_BOLD}System Requirements Validation:${RESET}"
  
  local requirements_passed=true

  # Docker check
  if [[ -n "$docker_ver" ]]; then
    echo -e "  [${GREEN}✓${RESET}] Docker Engine: v$docker_ver"
  else
    echo -e "  [${RED}✗${RESET}] Docker Engine: Not Found (Required to run Compose)"
    requirements_passed=false
  fi

  # Compose check
  if [[ -n "$compose_cmd" ]]; then
    echo -e "  [${GREEN}✓${RESET}] Docker Compose: Available ($compose_cmd)"
  else
    echo -e "  [${RED}✗${RESET}] Docker Compose: Not Found (Required to run Compose)"
    requirements_passed=false
  fi

  # RAM check
  if [[ "$ram_gb" -ge 1 ]]; then
    echo -e "  [${GREEN}✓${RESET}] System RAM: ${ram_gb} GB (Min 1 GB passed)"
  else
    echo -e "  [${RED}✗${RESET}] System RAM: ${ram_gb} GB (Failed - Minimum 1 GB required for dedup-only)"
    requirements_passed=false
  fi

  if ! $requirements_passed; then
    echo -e "\n${YELLOW}Warning: Some system requirements are not met.${RESET}"
    if ! confirm "Would you like to proceed with configuring templates anyway?" "n"; then
      echo "Aborting setup."
      exit 1
    fi
  fi

  # Step 1: Choose Hardware Target
  echo -e "\n${NC_BOLD}Step 1: Choose Hardware Target${RESET}"
  local target_hw="cpu"
  if $HAS_NVIDIA; then
    echo "  1) CPU (recommended for general hosts, ARM, Macs)"
    echo "  2) GPU (NVIDIA CUDA)"
    local hw_choice
    hw_choice=$(prompt_user "Select Target Hardware" "1")
    if [[ "$hw_choice" == "2" ]]; then
      target_hw="gpu"
    fi
  else
    echo -e "  No NVIDIA GPU detected. Target hardware set to ${GREEN}CPU${RESET}."
    target_hw="cpu"
  fi

  # Step 2: Choose ML Runtime & Auto-Set Execution Provider
  local target_runtime="torch"
  
  if [[ "$target_hw" == "gpu" ]]; then
    # GPU uses pre-baked PyTorch config in docker-compose.gpu.yml
    target_runtime="torch"
    echo -e "\n${NC_BOLD}Step 2: Choose ML Runtime${RESET}"
    echo -e "  GPU deployment uses PyTorch (torch) runtime (configured in Docker Compose)."
  else
    # CPU Target: Ask user for Torch or ONNX
    echo -e "\n${NC_BOLD}Step 2: Choose ML Runtime${RESET}"
    
    # Check if CPU is ARM (Mac M-series or Linux AArch64) to recommend ONNX
    local is_arm=false
    if [[ "$arch" == "arm64" || "$arch" == "aarch64" || "$os" == "Darwin" ]]; then
      is_arm=true
    fi

    if $is_arm; then
      echo "  1) PyTorch (torch)"
      echo -e "  2) ONNX (onnx - ${GREEN}Recommended for ARM/Apple Silicon${RESET})"
      local runtime_choice
      runtime_choice=$(prompt_user "Select Inference Runtime" "2")
      if [[ "$runtime_choice" == "2" || "$runtime_choice" == "onnx" ]]; then
        target_runtime="onnx"
      fi
    else
      echo "  1) PyTorch (torch - default)"
      echo "  2) ONNX (onnx)"
      local runtime_choice
      runtime_choice=$(prompt_user "Select Inference Runtime" "1")
      if [[ "$runtime_choice" == "2" || "$runtime_choice" == "onnx" ]]; then
        target_runtime="onnx"
      fi
    fi
  fi

  # Step 3: Choose Vision Profile (Filtered based on system memory)
  echo -e "\n${NC_BOLD}Step 3: Choose Vision Profile${RESET}"
  local target_profile="quality"

  if [[ "$target_hw" == "gpu" ]]; then
    # GPU uses pre-baked quality profile config in docker-compose.gpu.yml
    target_profile="quality"
    echo -e "  GPU deployment uses ${GREEN}quality${RESET} profile (configured in Docker Compose)."
  else
    # CPU: Dynamic filtering by RAM
    if [[ "$ram_gb" -ge 4 ]]; then
      echo "  1) quality     (SigLIP2 384px + NSFW + Relevance. Recommended. Needs >=4GB RAM)"
      echo "  2) standard    (SigLIP2 224px + NSFW + Relevance. Fast. Needs >=3GB RAM)"
      echo "  3) dedup-only  (No ML models, simple duplicate checks. Fast. Needs >=512MB RAM)"
      local profile_choice
      profile_choice=$(prompt_user "Select Vision Profile" "1")
      if [[ "$profile_choice" == "2" ]]; then
        target_profile="standard"
      elif [[ "$profile_choice" == "3" ]]; then
        target_profile="dedup-only"
      fi
    elif [[ "$ram_gb" -ge 3 ]]; then
      echo -e "  ${YELLOW}Notice: quality profile hidden due to low system RAM (<4 GB)${RESET}"
      echo "  1) standard    (SigLIP2 224px + NSFW + Relevance. Fast. Needs >=3GB RAM)"
      echo "  2) dedup-only  (No ML models, simple duplicate checks. Fast. Needs >=512MB RAM)"
      local profile_choice
      profile_choice=$(prompt_user "Select Vision Profile" "1")
      if [[ "$profile_choice" == "2" ]]; then
        target_profile="dedup-only"
      else
        target_profile="standard"
      fi
    else
      echo -e "  Low system RAM (${ram_gb} GB < 3 GB). Vision profiles hidden."
      echo -e "  Automatically set profile to ${GREEN}dedup-only${RESET}."
      target_profile="dedup-only"
    fi
  fi

  # Step 4: Configure Redis Connection (URI)
  echo -e "\n${NC_BOLD}Step 4: Configure Redis Connection${RESET}"
  echo -e "  Please specify the Redis connection URI for the worker."
  echo -e "  ${NC_DIM}Example (local dev):  redis://host.docker.internal:6379${RESET}"
  echo -e "  ${NC_DIM}Example (managed):    rediss://default:your-token@your-host.upstash.io:6379${RESET}"
  
  local target_redis_url
  target_redis_url=$(prompt_user "Enter Redis URI" "redis://host.docker.internal:6379")

  # Overwrite Warnings
  if [[ -f "docker-compose.yml" ]]; then
    echo -e "${YELLOW}Warning: docker-compose.yml already exists in the current directory.${RESET}"
    if ! confirm "Are you sure you want to overwrite it?" "n"; then
      echo "Aborting setup to protect existing files."
      exit 0
    fi
  fi

  if [[ -f ".env" ]]; then
    echo -e "${YELLOW}Warning: .env already exists in the current directory.${RESET}"
    if ! confirm "Are you sure you want to overwrite it?" "n"; then
      echo "Aborting setup to protect existing files."
      exit 0
    fi
  fi

  # Setup Files (Download or Copy)
  echo -e "\n${NC_BOLD}Step 5: Setting up configurations...${RESET}"
  
  if $IS_CLONED; then
    # We are in a cloned repo, we can copy locally
    if [[ "$target_hw" == "gpu" ]]; then
      cp docker-compose.gpu.yml docker-compose.yml
      echo "Copied docker-compose.gpu.yml -> docker-compose.yml"
    else
      # If they are inside the repo and want CPU, restore default CPU docker-compose.yml if modified
      if git diff --name-only | grep -q "docker-compose.yml" 2>/dev/null; then
        echo "Local docker-compose.yml has modifications. Restoring default CPU version from Git..."
        git checkout -- docker-compose.yml 2>/dev/null || true
      fi
    fi
    cp .env.example .env
    echo "Copied .env.example -> .env"
  else
    # Running standalone, download from GitHub
    if [[ "$target_hw" == "gpu" ]]; then
      download_file "${GITHUB_RAW_URL}/docker-compose.gpu.yml" "docker-compose.yml"
    else
      download_file "${GITHUB_RAW_URL}/docker-compose.yml" "docker-compose.yml"
    fi
    download_file "${GITHUB_RAW_URL}/.env.example" ".env"
    echo "Downloaded templates from GitHub."
  fi

  # Apply Environment Variables to .env
  update_env_var "REDIS_URL" "$target_redis_url" ".env"
  update_env_var "VISION_PROFILE" "$target_profile" ".env"
  update_env_var "INFERENCE_RUNTIME" "$target_runtime" ".env"

  if [[ "$target_hw" == "gpu" ]]; then
    update_env_var "DEVICE" "cuda" ".env"
    if [[ "$target_runtime" == "onnx" ]]; then
      update_env_var "ORT_EXECUTION_PROVIDER" "cuda" ".env"
    fi
  else
    update_env_var "DEVICE" "cpu" ".env"
    if [[ "$target_runtime" == "onnx" ]]; then
      update_env_var "ORT_EXECUTION_PROVIDER" "cpu" ".env"
    fi
  fi

  echo -e "${GREEN}Configuration generated successfully!${RESET}"
  echo -e "  - ${NC_BOLD}docker-compose.yml${RESET} is configured for ${NC_BOLD}${target_hw}${RESET}"
  echo -e "  - ${NC_BOLD}.env${RESET} is generated with profile: ${NC_BOLD}${target_profile}${RESET}, runtime: ${NC_BOLD}${target_runtime}${RESET}"
  echo -e "  - Redis URL is configured as: ${NC_BOLD}${target_redis_url}${RESET}"

  # Offer to run immediately
  if [[ -n "$compose_cmd" ]]; then
    if confirm "Would you like to start the Docker Compose worker now?" "y"; then
      echo "Starting worker..."
      $compose_cmd up -d
      echo -e "${GREEN}Worker started! Run '$compose_cmd ps' to verify state or '$compose_cmd logs -f' for logs.${RESET}"
    else
      echo -e "To start later, run: ${NC_BOLD}$compose_cmd up -d${RESET}"
    fi
  else
    echo -e "To start the worker, install Docker and run: ${NC_BOLD}docker compose up -d${RESET}"
  fi
}

# --- Main Entry Point ---

# Check if we are inside a cloned repository
IS_CLONED=false
ROOT_DIR=""

# Try finding directory context
if [[ -f "$0" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  if [[ -f "$SCRIPT_DIR/../pyproject.toml" && -d "$SCRIPT_DIR/../app" ]]; then
    IS_CLONED=true
    ROOT_DIR="$SCRIPT_DIR/.."
  fi
fi

if ! $IS_CLONED; then
  if [[ -f "pyproject.toml" && -d "app" ]]; then
    IS_CLONED=true
    ROOT_DIR="$(pwd)"
  fi
fi

# If inside clone, move to repo root
if $IS_CLONED; then
  cd "$ROOT_DIR"
fi

# Parse subcommands
if [[ $# -eq 0 ]]; then
  # No arguments provided
  if $IS_CLONED; then
    echo -e "UnLostPaws Vision Worker CLI"
    echo -e "Usage: ./tools/run.sh <subcommand> [options]"
    echo -e "       Subcommands: doctor, smoke, benchmark, export, validate, setup\n"
    
    if confirm "No subcommand specified. Would you like to run the interactive setup wizard?" "y"; then
      run_setup
    else
      exit 0
    fi
  else
    # Standalone script run (e.g. curl | bash)
    run_setup
  fi
  exit 0
fi

SUBCOMMAND="$1"
shift

case "$SUBCOMMAND" in
  setup)
    run_setup
    ;;
  doctor|smoke|benchmark|export|validate)
    if ! $IS_CLONED; then
      echo -e "${RED}Error: Subcommand '$SUBCOMMAND' is only available inside a cloned repository.${RESET}" >&2
      echo -e "To use CLI subcommands, clone the repository first:" >&2
      echo -e "  git clone https://github.com/the-dot-squad/unlostpaws-worker.git" >&2
      exit 1
    fi
    
    # Locate Python interpreter
    if [[ -x ".venv/bin/python" ]]; then
      PYTHON=".venv/bin/python"
    elif command -v python3.12 >/dev/null 2>&1; then
      PYTHON="python3.12"
    else
      PYTHON="${PYTHON:-python3}"
    fi
    
    exec "$PYTHON" -m tools "$SUBCOMMAND" "$@"
    ;;
  help|--help|-h)
    echo -e "UnLostPaws Vision Worker CLI"
    echo -e "Usage: ./tools/run.sh <subcommand> [options]"
    echo -e "       ./tools/run.sh setup"
    echo -e "\nAvailable Subcommands:"
    echo -e "  setup        - Run interactive Docker Compose setup wizard (default)"
    echo -e "  doctor       - Run preflight hardware and profile validation check"
    echo -e "  smoke        - Run pipeline smoke checks"
    echo -e "  benchmark    - Profile performance of different model runtimes"
    echo -e "  export       - Export PyTorch model weights to ONNX format"
    echo -e "  validate     - Validate inputs or dataset records"
    ;;
  *)
    echo -e "${RED}Error: Unknown subcommand '$SUBCOMMAND'${RESET}" >&2
    echo -e "Run './tools/run.sh --help' for usage." >&2
    exit 1
    ;;
esac
