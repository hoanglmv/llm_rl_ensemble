#!/bin/bash

# --- Cấu hình ---
INSTALL_DIR="$HOME/miniconda3"
MINICONDA_URL_BASE="https://repo.anaconda.com/miniconda"

# Màu sắc cho log dễ nhìn
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INSTALLER] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# 1. Kiểm tra xem Conda đã cài chưa
if [ -d "$INSTALL_DIR" ]; then
    log "Conda đã được cài đặt tại $INSTALL_DIR."
    log "Bỏ qua quá trình cài đặt."
    exit 0
fi

# 2. Phát hiện Hệ điều hành (Linux hoặc MacOS)
OS_NAME=$(uname -s)
if [ "$OS_NAME" == "Linux" ]; then
    M_OS="Linux"
elif [ "$OS_NAME" == "Darwin" ]; then
    M_OS="MacOSX"
else
    error "Hệ điều hành không được hỗ trợ: $OS_NAME. Script này chỉ chạy trên Linux hoặc macOS."
fi

# 3. Phát hiện Kiến trúc CPU (x86_64 hoặc ARM64/aarch64)
ARCH_NAME=$(uname -m)
if [ "$ARCH_NAME" == "x86_64" ]; then
    M_ARCH="x86_64"
elif [ "$ARCH_NAME" == "aarch64" ]; then
    M_ARCH="aarch64" # Thường dùng cho Linux ARM
elif [ "$ARCH_NAME" == "arm64" ]; then
    M_ARCH="arm64"   # Thường dùng cho macOS Apple Silicon
else
    error "Kiến trúc CPU không được hỗ trợ: $ARCH_NAME"
fi

# Xử lý đặc biệt: Anaconda đặt tên file cho Mac ARM là 'arm64' nhưng Linux ARM là 'aarch64'
# Script trên đã cover, nhưng cần đảm bảo đúng format link download.

INSTALLER_NAME="Miniconda3-latest-${M_OS}-${M_ARCH}.sh"
DOWNLOAD_URL="${MINICONDA_URL_BASE}/${INSTALLER_NAME}"

log "Phát hiện hệ thống: $M_OS ($M_ARCH)"
log "Đang chuẩn bị tải: $INSTALLER_NAME"

# 4. Tải bộ cài (Sử dụng curl hoặc wget)
if command -v curl &> /dev/null; then
    curl -L -O "$DOWNLOAD_URL"
elif command -v wget &> /dev/null; then
    wget "$DOWNLOAD_URL"
else
    error "Không tìm thấy 'curl' hoặc 'wget'. Vui lòng cài đặt trước khi chạy script."
fi

if [ ! -f "$INSTALLER_NAME" ]; then
    error "Tải xuống thất bại."
fi

# 5. Cài đặt (Chế độ batch - không cần user tương tác)
log "Đang cài đặt Miniconda vào $INSTALL_DIR..."
bash "$INSTALLER_NAME" -b -p "$INSTALL_DIR"

if [ $? -ne 0 ]; then
    error "Cài đặt thất bại."
fi

# 6. Dọn dẹp file cài đặt
rm "$INSTALLER_NAME"

# 7. Khởi tạo Conda cho Shell hiện tại (Bash/Zsh)
log "Đang khởi tạo shell..."
eval "$($INSTALL_DIR/bin/conda shell.bash hook)"

# Init cho shell mặc định của user
if [ -n "$ZSH_VERSION" ]; then
    "$INSTALL_DIR/bin/conda" init zsh
elif [ -n "$BASH_VERSION" ]; then
    "$INSTALL_DIR/bin/conda" init bash
else
    "$INSTALL_DIR/bin/conda" init
fi

log "Cài đặt hoàn tất!"
log "Vui lòng tắt terminal và mở lại, hoặc chạy lệnh sau để sử dụng ngay:"
echo -e "${GREEN}source ~/.bashrc${NC