### Data

To recreate s1K follow the steps below. In various files you will have to rename the organizations `simplescaling` and `qfq` with an organization that you own. **Note that [s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1) is a better dataset generated with r1 traces instead of Gemini traces.**
1. Run `data/collect_data.py` followed by `data/fix_gpqa.py` & `data/add_aime.py` to collect the questions; Make sure to change the hub path in the respective files to one of your own.
2. Generate traces with Gemini via `python data/gemini.py`. This step will use https://hf.co/datasets/qfq/train which should be roughly equivalent to the dataet you have produced in 1.
3. Generate answers with Qwen via `python data/bulk_inference.py` that can be launched with `data/bulk_inference.sh`.
4. Add features by running `python data/featurization.py`.
5. Run final filtering via going through `data/filter.ipynb`.