The `vce_for_videomae` directory contains an adapter to the main VCE dataset, in the CSV format expected by the VideoMAE finetuning script.

The default adapter assumes that your main VCE dataset is located at `/data/vce_dataset/`. If this does not match your own VCE dataset path, you can re-run the generation script to re-generate the CSV's to match own dataset path:
```
python vce_to_videomae_dataset.py -d /PATH/TO/vce_dataset/
```

> Note that this adapter is not required for the V2V dataset, as the V2V finetuning script works directly on the unmodified V2V dataset.