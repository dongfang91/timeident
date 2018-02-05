# timeident

Python version: 2.7.13
Python Package: h5py, nltk 3.2.4, anaforatools, keras 1.2.1 , theano 0.9.0rc2.dev-674f8d5148cb8dcea446e7243e629d8f83647da5

To processe the documents, please run:
python preprocess.py --raw "data/THYMEColonFinal/Dev" --out "data/dev/Dev1" --processed "true" --file "file_name.txt"

To postprocess keras outputs into anafora format:
python output.py --raw "data/THYMEColonFinal/Dev" --preocessed_path "data/dev/Dev1" --model "weights-improvement-685.hdf5" --out "output_pred_path"


