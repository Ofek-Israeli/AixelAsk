KCONFIG      := Kconfig
DOTCONFIG    := .config
DEFCONFIG    := defconfig
PYTHON       := python3
SRC          := src
MENUCONFIG   := tools/menuconfig.py

.PHONY: menuconfig defconfig run server-start server-stop download test clean clean-all \
       run-baseline run-baseline-fewshot-ph-ext \
       train train-grpo train-overfit-poc resume-train \
       test-trained test-trained-best test-trained-latest \
       compile-curves show-train-config show-paths

menuconfig: $(MENUCONFIG)
	@$(PYTHON) -c "import kconfiglib" 2>/dev/null || \
		{ echo "ERROR: kconfiglib not installed. Run: pip install kconfiglib"; exit 1; }
	KCONFIG_CONFIG=$(DOTCONFIG) $(PYTHON) $(MENUCONFIG) $(KCONFIG)

defconfig:
	cp $(DEFCONFIG) $(DOTCONFIG)

run: $(DOTCONFIG)
	$(PYTHON) -m src.main --config $(DOTCONFIG)

server-start: $(DOTCONFIG)
	$(PYTHON) -m src.sglang_server --config $(DOTCONFIG) --action start

server-stop:
	$(PYTHON) -m src.sglang_server --action stop

download: $(DOTCONFIG)
	$(PYTHON) -m src.download_models --config $(DOTCONFIG)

test:
	$(PYTHON) -m pytest tests/ -v

run-baseline: $(DOTCONFIG)
	$(PYTHON) -m src.main --config $(DOTCONFIG) \
		--override CONFIG_DAG_PROMPT_VARIANT=DAG_PROMPT_MIN_DEPTH_BASELINE \
		--override CONFIG_RESULT_FILE=output/results_min_depth.jsonl \
		--override CONFIG_DAG_STATS_FILE=output/dag_stats_min_depth.json

run-baseline-fewshot-ph-ext: $(DOTCONFIG)
	$(PYTHON) -m src.main --config $(DOTCONFIG) \
		--override CONFIG_FEWSHOT_VARIANT=FEWSHOT_PARALLEL_HYBRID_EXTENDED \
		--override CONFIG_RESULT_FILE=output/results_fewshot_ph_ext.jsonl \
		--override CONFIG_DAG_STATS_FILE=output/dag_stats_fewshot_ph_ext.json

train: $(DOTCONFIG)
	$(PYTHON) -m src.training.train_main --config $(DOTCONFIG)

train-grpo:
	cp defconfig.train_grpo $(DOTCONFIG)
	$(PYTHON) -m src.training.train_main --config $(DOTCONFIG)

train-overfit-poc:
	cp defconfig.train_overfit_poc $(DOTCONFIG)
	$(PYTHON) -m src.training.train_main --config $(DOTCONFIG)

resume-train: $(DOTCONFIG)
	$(PYTHON) -m src.training.train_main --config $(DOTCONFIG) --resume

test-trained: $(DOTCONFIG)
	$(PYTHON) -m src.training.test_main --config $(DOTCONFIG)

test-trained-best: $(DOTCONFIG)
	$(PYTHON) -m src.training.test_main --config $(DOTCONFIG) --checkpoint best

test-trained-latest: $(DOTCONFIG)
	$(PYTHON) -m src.training.test_main --config $(DOTCONFIG) --checkpoint latest

compile-curves: $(DOTCONFIG)
	$(PYTHON) -m src.training.tex_compile --config $(DOTCONFIG) --all

show-train-config: $(DOTCONFIG)
	@grep -E '^CONFIG_(TRAIN|GRPO|REWARD|OVERFIT|GLOBAL_SEED|TRAINING|SPLIT|TEST_TRAINED)' $(DOTCONFIG) | sort

show-paths: $(DOTCONFIG)
	@$(PYTHON) -c "import os; from src.config import load_config; from src.download_models import resolve_model_path; c = load_config('$(DOTCONFIG)'); print('PROJECT_DIR (immutable module root):', c.PROJECT_DIR); print('AIXELASK_ROOT (upstream repo root):', c.AIXELASK_ROOT); print('UPSTREAM_SOURCE_ROOT (upstream Python src):', c.UPSTREAM_SOURCE_ROOT); print('UPSTREAM_SCRIPTS_DIR (upstream scripts/):', c.UPSTREAM_SCRIPTS_DIR); print('PERSISTENT_ROOT (output/cache root):', c.PERSISTENT_ROOT); print('MODEL_CACHE_DIR (HF_HOME):', c.MODEL_CACHE_DIR); print('HUB_CACHE (actual model storage):', os.path.join(c.MODEL_CACHE_DIR, 'hub')); print('RESOLVED_MODEL_PATH (local snapshot):', resolve_model_path(c)); print('EMBEDDING_CACHE:', c.EMBEDDING_CACHE); print('RESULT_FILE:', c.RESULT_FILE); print('LOG_FILE:', c.LOG_FILE); print('DAG_STATS_FILE:', c.DAG_STATS_FILE); print('TRAIN_OUTPUT_DIR:', c.TRAIN_OUTPUT_DIR); print('TRAIN_CURVES_DIR:', c.TRAIN_CURVES_DIR); print('EPHEMERAL_TMPDIR:', c.EPHEMERAL_TMPDIR); print('SPLIT_MODE:', c.SPLIT_MODE); print('INFERENCE_DATASET_PATH:', c.INFERENCE_DATASET_PATH); print('HF_HOME (set by config.py):', os.environ.get('HF_HOME', 'not set')); print('TRANSFORMERS_CACHE (set by config.py):', os.environ.get('TRANSFORMERS_CACHE', 'not set'))"

clean:
	rm -f $(DOTCONFIG)
	rm -rf __pycache__ src/__pycache__ tests/__pycache__

clean-all:
	@echo "WARNING: This deletes project-local persistent artifacts."
	@echo "HF model cache (CONFIG_MODEL_CACHE_DIR) is NOT deleted."
	@echo "Press Ctrl+C within 5s to abort." && sleep 5
	@$(PYTHON) -c "from src.config import load_config; import shutil, os; c = load_config('$(DOTCONFIG)') if os.path.exists('$(DOTCONFIG)') else None; \
		[shutil.rmtree(p, ignore_errors=True) for p in ['output/', 'cache/'] + ([os.path.dirname(c.RESULT_FILE), os.path.dirname(c.EMBEDDING_CACHE), c.TRAIN_OUTPUT_DIR] if c else [])]" 2>/dev/null || \
		rm -rf output/ cache/
	rm -f $(DOTCONFIG)
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -f .sglang.pid
