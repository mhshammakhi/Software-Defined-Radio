# Root Makefile for Software-Defined-Radio.
#
# `make gpu-version` detects the SM architecture of the GPU installed on
# this machine (via nvidia-smi) and stages the pre-built binary for the
# nearest available sm version <= the detected one from GPU_Based/<sm_xx>/
# into the root build/ directory, alongside the config.ini and data/ it
# needs to run.

BUILD_DIR     := build
GPU_BASED_DIR := GPU_Based

CONFIG_SRC := $(GPU_BASED_DIR)/config.ini
DATA_SRC   := $(GPU_BASED_DIR)/data
OUTPUT_CHECK_SCRIPT := $(GPU_BASED_DIR)/scripts/output_check.py

.PHONY: gpu-version clean

gpu-version:
	@cap=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' '); \
	if [ -z "$$cap" ]; then \
		echo "Error: could not read GPU compute capability from nvidia-smi. Is a CUDA GPU installed?" >&2; \
		exit 1; \
	fi; \
	detected=$$(echo $$cap | tr -d '.'); \
	best=""; best_num=-1; \
	for d in $(GPU_BASED_DIR)/sm_*; do \
		[ -d "$$d" ] || continue; \
		name=$$(basename "$$d"); \
		num=$${name#sm_}; \
		case $$num in ''|*[!0-9]*) continue;; esac; \
		if [ "$$num" -le "$$detected" ] && [ "$$num" -gt "$$best_num" ]; then \
			best_num=$$num; best=$$name; \
		fi; \
	done; \
	if [ -z "$$best" ]; then \
		echo "Error: no pre-built sm version <= detected sm_$$detected found under $(GPU_BASED_DIR)/." >&2; \
		exit 1; \
	fi; \
	arch=$$best; \
	src_dir="$(GPU_BASED_DIR)/$$arch"; \
	echo "==> Detected GPU compute capability $$cap (sm_$$detected) -> nearest available $$arch, using $$src_dir"; \
	mkdir -p $(BUILD_DIR); \
	cp -f "$$src_dir/sdr_gpu" $(BUILD_DIR)/; \
	cp -f $(CONFIG_SRC) $(BUILD_DIR)/; \
	rm -rf $(BUILD_DIR)/data; \
	cp -r $(DATA_SRC) $(BUILD_DIR)/data; \
	cp -f $(OUTPUT_CHECK_SCRIPT) $(BUILD_DIR)/; \
	echo "==> build/ ready with sdr_gpu ($$arch), config.ini, and data/"

clean:
	rm -rf $(BUILD_DIR)
