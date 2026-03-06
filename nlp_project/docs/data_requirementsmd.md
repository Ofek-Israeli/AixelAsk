# Data Requirements for Mistral-7B GRPO+LoRA

In RLHF and reasoning tasks, prior work uses **tens of thousands** of examples (prompts or comparisons).  For instance, *InstructGPT* (Ouyang *et al.*, 2022) reports using about **13K** prompts for supervised fine-tuning, **33K** for reward-model training, and **31K** for PPO (RLHF) training【76†L1-L4】【77†L1-L4】.  Similarly, Lambert *et al.* (2024) describe a modern instruction-tuning recipe (“Tülu 3”) where the final RL stage uses only **∼10K** prompts to boost core skills like math【61†L323-L330】.  In that recipe, a 10K-prompt RL run is treated as a “small-scale” fine-tuning stage for reasoning.  Correspondingly, Shao *et al.* (2024, DeepSeek R1) employ a **cold-start** of **100K+** on-policy reasoning examples, followed by large-scale RL【61†L341-L349】.  This suggests **100K+** is used in very large-scale runs, whereas 10K–30K is common for moderate experiments.  

Data-efficient methods also support the ~10K scale.  Lai *et al.* (2024) introduce **Step-DPO** for math reasoning using *stepwise* preference pairs.  They collect **~10K** step-by-step preference samples and show that even **10K pairs** (≲500 training steps) suffice to improve  math accuracy by ~3%【79†L49-L57】.  This indicates that for complex reasoning tasks, on the order of ten thousand supervised examples can yield measurable gains.  

**Recommendation:**  Given these precedents and a single RTX 5090 (32 GB) with LoRA, we suggest **on the order of 10K prompts** for an initial RLHF run.  For example, sampling *M* completions per prompt (say *M*≈8) means ∼80K generated sequences.  With mixed precision and 4-bit weights, this would require a feasible amount of GPU time (on the order of hours to days).  One could start at **5–10K prompts** (with group size 8 giving ~40–80K samples) as a practical minimum, then scale up if needed.  Note that Ouyang *et al.* (2022) found 13K–31K prompts sufficient for InstructGPT on GPT-3【76†L1-L4】【77†L1-L4】, and Lambert *et al.* (2024) use ~10K for a small RL stage【61†L323-L330】.  Thus, **~10K** is a defensible lower bound. If resources allow, a larger set (20K–30K) would match larger prior runs; DeepSeek R1 used **~100K** for maximal performance【61†L341-L349】.  

**Summary of Example Counts from Literature:**  
- **Ouyang *et al.*, 2022:** SFT on ~13K prompts; RM on ~33K; PPO (RL) on ~31K【76†L1-L4】【77†L1-L4】.  
- **Lambert *et al.*, 2024:** RLHF math stage on ~10K prompts (small-scale run)【61†L323-L330】.  
- **Lai *et al.*, 2024 (Step-DPO):** ~10K preference pairs suffice for math reasoning improvement【79†L49-L57】.  
- **Shao *et al.*, 2024 (DeepSeek R1):** “Cold-start” with 100K+ on-policy reasoning examples【61†L341-L349】.  

In sum, a **few×10^4 examples** is typical for these tasks.  For Mistral-7B with LoRA on one 5090 GPU, we recommend starting with on the order of **5K–10K training prompts**, scaling upward as resources and convergence demand.  

**Sources:** Ouyang *et al.* (2022)【76†L1-L4】【77†L1-L4】; Lambert *et al.* (2024)【61†L323-L330】; Lai *et al.* (2024)【79†L49-L57】; Shao *et al.* (2024)【61†L341-L349】. (APA-style)