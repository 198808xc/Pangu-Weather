## Pangu-Weather

This is the official repository for the Pangu-Weather papers.

[Accurate medium-range global weather forecasting with 3D neural networks](https://www.nature.com/articles/s41586-023-06185-3), Nature, Volume 619, Pages 533–538, 2023.

[Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast](https://arxiv.org/abs/2211.02556), arXiv preprint: 2211.02556, 2022.

*by Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu and Qi Tian*

**Note: the arXiv version offers more technical details, and the Nature paper contains some new figures.**

Resources including pseudocode, pre-trained models, and inference code are released here.

The slides used in a series of recent talks are attached here. [Baidu Netdisk](https://pan.baidu.com/s/1eYkmf0QehEYdR1dk12idNA?pwd=zjj4), extraction code: zjj4

中文版PPT请参见链接: [百度网盘](https://pan.baidu.com/s/1oINuxuBstFEbfcouX6qHMw?pwd=7nzb), 提取码: 7nzb

## News and Updates

* [Jul 31 2023] We released the details of training the lite version of Pangu-Weather.
* [Jul 19 2023] ECMWF released an official [technical report](https://arxiv.org/abs/2307.10128) for "the rise of data-driven weather forecasting". Pangu-Weather was mentioned and tested thoroughly in the paper. We thank ECMWF for testing our models in real-world scenarios.
* [Jul 17 2023] Pangu-Weather was online as part of ECMWF's operational suite! Everyone can see 10-day global weather forecasting **without running code**. ECMWF has made use of the models released at this repository! [Please search the ECMWF charts website with the query of "PANGU".](https://charts.ecmwf.int/?query=pangu)
* [Jul 05 2023] Pangu-Weather was published on [Nature](https://www.nature.com/articles/s41586-023-06185-3). It was made **Open Access**! We recommend the researchers to cite our Nature paper in the future.
* [Jun 27 2023] Pangu-Weather was presented at [PASC 2023](https://pasc23.pasc-conference.org/program/schedule/).
* [Jun 12 2023] Pangu-Weather was presented at [VALSE 2023](http://valser.org/2023/#/workshopde?id=15).
* [May 27 2023] Pangu-Weather was presented at [the WMO Early Warning for All (EW4ALL) conference](https://community.wmo.int/en/news/exploring-possibilities-artificial-intelligence-areas-water-weather-and-climate).
* [May 12 2023] ECMWF released a [repository](https://github.com/ecmwf-lab/ai-models-panguweather), offering a toolkit for running Pangu-Weather. We thank ECMWF for the efforts in easing everyone to test Pangu-Weather.
* [May 09 2023] Pangu-Weather was accepted by Nature!

## Installation

The downloaded files shall be organized as the following hierarchy:

```plain
├── root
│   ├── input_data
│   │   ├── input_surface.npy
│   │   ├── input_upper.npy
│   ├── output_data
│   ├── pangu_weather_1.onnx
│   ├── pangu_weather_3.onnx
│   ├── pangu_weather_6.onnx
│   ├── pangu_weather_24.onnx
│   ├── inference_cpu.py
│   ├── inference_gpu.py
│   ├── inference_iterative.py
```

If you use a CPU environment, please run:
```
pip install -r requirements_cpu.txt
```

If you use a GPU environment, please first confirm that the cuda version is 11.6 and the cudnn version is the 8.2.4 for Linux and 8.5.0.96 for Windows (please see [this page](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for details). Then, please run:
```
pip install -r requirements_gpu.txt
```

## Global weather forecasting (inference) using the trained models

#### Downloading trained models

Please download the four pre-trained models (~1.1GB each) from Google drive or Baidu netdisk:

The 1-hour model (pangu_weather_1.onnx): [Google drive](https://drive.google.com/file/d/1fg5jkiN_5dHzKb-5H9Aw4MOmfILmeY-S/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1M7SAigVsCSH8hpw6DE8TDQ?pwd=ie0h)

The 3-hour model (pangu_weather_3.onnx): [Google drive](https://drive.google.com/file/d/1EdoLlAXqE9iZLt9Ej9i-JW9LTJ9Jtewt/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/197fZsoiCqZYzKwM7tyRrfg?pwd=gmcl)

The 6-hour model (pangu_weather_6.onnx): [Google drive](https://drive.google.com/file/d/1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1q7IB7tNjqIwoGC7KVMPn4w?pwd=vxq3)

The 24-hour model (pangu_weather_24.onnx): [Google drive](https://drive.google.com/file/d/1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/179q2gkz2BrsOR6g3yfTVQg?pwd=eajy)

These models are stored using the ONNX format, and thus can be used via different languages such as Python, C++, C#, Java, etc.

#### Input data preparation using Python

Please prepare the input data using [numpy](https://numpy.org/). There are two files that shall be put under the `input_data` folder, namely, `input_surface.npy` and `input_upper.npy`.

`input_surface.npy` stores the input surface variables. It is a numpy array shaped (4,721,1440) where the first dimension represents the 4 surface variables (MSLP, U10, V10, T2M **in the exact order**).

`input_upper.npy` stores the upper-air variables. It is a numpy array shaped (5,13,721,1440) where the first dimension represents the 5 surface variables (Z, Q, T, U and V **in the exact order**), and the second dimension represents the 13 pressure levels (1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa, 300hPa, 250hPa, 200hPa, 150hPa, 100hPa and 50hPa **in the exact order**).

In both cases, the dimensions of 721 and 1440 represent the size along the latitude and longitude, where the numerical range is [90,-90] degree and [0,359.75] degree, respectively, and the spacing is 0.25 degrees. For each 721x1440 slice, the data format is exactly the same as the `.nc` file download from the ERA5 official website.

Note that the numpy arrays should be in single precision (`.astype(np.float32)`), not in double precision.

We support ERA5 initial fields and ECMWF initial fields (e.g., the initial fields of the HRES forecast), where the latter often leads to a slight accuracy drop (mainly for T2M because the two fields are quite different in temperature). A `.nc` file of ERA5 can be transformed into a `.npy` file using the netCDF4 package, and a `.grib` file of the ECMWF initial fields can be transformed into a `.npy` file using the pygrib package. Note that Z represents geopotential, not geopotential height, so a factor of 9.80665 should be multiplied if the raw data contains the geopotential height.

We temporarily do not support other kinds of initial fields due to the possibly dramatic differences in the fields when Z<0.

We provide an example of transferred input files, `input_surface.npy` and `input_upper.npy`, which correspond to the ERA5 initial fields of at 12:00UTC, 2018/09/27. Please download them from Google drive or Baidu netdisk:

`input_surface.npy`: [Google drive](https://drive.google.com/file/d/1pj8QEVNpC1FyJfUabDpV4oU3NpSe0BkD/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1i4o5i8guAqmOus6PWncAlA?pwd=4z9s)

`input_upper.npy`: [Google drive](https://drive.google.com/file/d/1--7xEBJt79E3oixizr8oFmK_haDE77SS/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1mS8X5MqEdbVfF2u2Us62FQ?pwd=sgx6)

#### Inference

After the above steps are finished, please check `inference_cpu.py` for an example of making a 24-hour weather forecast on CPU with the 24-hour model, and `inference_gpu.py` for the GPU version.

For example, running the following command, one can get the 24-hour forecast in the `output_data` folder:
```
python inference_cpu.py # python inference_gpu.py for gpu environment
```

Also, `inference_iterative.py` shows an example to generate per-6-hour forecast within a week.

## Pseudocode and how to use

`pseudocode.py` contains the pseudocode that elaborates our main algorithm. It is written in Python and can be implemented using any deep learning library, e.g. PyTorch and TensorFlow.

Note that one needs to download about 60TB of ERA5 data and prepare for computational resource of 3000 GPU-days (in V100) to train each model.

## Training a lite version

Recently, we found that Pangu-Weather can be trained efficiently using only 1% of data and GPU computation. We call the version Pangu-Weather-lite. Note that the lite models cannot rival the full models, but the lite version offers opportunities for researchers with limited resource to explore the AI methods for weather forecasting.

Here are the key implementation details.

* Data. We reduced the training data into 11 years (2007-2017) and only used the 00UTC time point (the full version used all 24 time points throughout the day). Also, only 00UTC data is used in the testing phase. The total amount of downloaded data shall be less than 1TB.
* Model. We adjusted the down-sampling rate in the first stage from 2x4x4 to 2x8x8.
* Training epochs. One can remain using 100 epochs or reduce the number to 50 (half); note that the cosine annealing schedule is adjusted accordingly.
* Model set. We only trained one model (lead time is 24 hours), which means that the lite version can only perform daily weather forecasting.

Here are the results.

| Method              | RMSE, Z500             | RMSE, T850           | RMSE, T2M            | RMSE, U10            | Years | Down-sampling | Epochs | GPU x days |
| ------------------- | ---------------------- | -------------------- | -------------------- | -------------------- | ----- | ------------- | --     | ---------- |
| Operational IFS     | 152.8 (3d), 333.7 (5d) | 1.37 (3d), 2.06 (5d) | 1.34 (3d), 1.75 (5d) | 1.94 (3d), 2.90 (5d) | --    | --            | --     | --         |
| Pangu-Weather       | 134.5 (3d), 296.7 (5d) | 1.14 (3d), 1.79 (5d) | 1.05 (3d), 1.53 (5d) | 1.61 (3d), 2.53 (5d) | 39    | 2 x 4 x 4     | 100    | 192 x 16   |
| Pangu-Weather-Lite1 | 163.1 (3d), 338.2 (5d) | 1.29 (3d), 1.96 (5d) | 1.16 (3d), 1.64 (5d) | 1.80 (3d), 2.74 (5d) | 11    | 2 x 8 x 8     | 100    | 8 x 6      |
| Pangu-Weather-Lite2 | 177.9 (3d), 357.5 (5d) | 1.36 (3d), 2.05 (5d) | 1.24 (3d), 1.71 (5d) | 1.90 (3d), 2.84 (5d) | 11    | 2 x 8 x 8     | 50     | 8 x 3      |

One can observe that the lite version can surpass operational IFS (*when tested only at 00UTC time points*) in T850 (850hPa temperature), T2M (2m temperature) and U10 (u-component of 10m wind speed), while requiring less than 1% of computational costs compared to the full version.

Please note that the lite version was only trained and tested in 00UTC data. This means that its performance on other time points is not guaranteed. Since whether variables are closely correlated to time-in-day, it is difficult to directly use the lite version for daily whether forecasting. Again, the lite version is to ease the researchers to explore the property of AI models.

## License

Pangu-Weather was released by Huawei Cloud.

The trained parameters of Pangu-Weather were made available under the terms of the BY-NC-SA 4.0 license. You can find details [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**The commercial use of these models is forbidden.**

Also, please note that all models were trained using the ERA5 dataset provided by ECMWF. Please do follow [their policy](https://apps.ecmwf.int/datasets/licences/copernicus/).

## References

If you use the resource in your research, please cite our paper:

```
@article{bi2023accurate,
  title={Accurate medium-range global weather forecasting with 3D neural networks},
  author={Bi, Kaifeng and Xie, Lingxi and Zhang, Hengheng and Chen, Xin and Gu, Xiaotao and Tian, Qi},
  journal={Nature},
  volume={619},
  number={7970},
  pages={533--538},
  year={2023},
  publisher={Nature Publishing Group}
}
```

We also offer the bibliography of the arXiv preprint version for your information.

```
@article{bi2022pangu,
  title={Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast},
  author={Bi, Kaifeng and Xie, Lingxi and Zhang, Hengheng and Chen, Xin and Gu, Xiaotao and Tian, Qi},
  journal={arXiv preprint arXiv:2211.02556},
  year={2022}
}
```
