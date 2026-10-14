"""run_infer.py

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlaid color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC, 
                              'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size per 1 GPU. [default: 32]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""
"""run_infer.py

用法:
  run_infer.py [选项] [--help] <命令> [<参数>...]
  run_infer.py --version
  run_infer.py (-h | --help)

选项:
  -h --help                   显示此帮助信息。
  --version                   显示版本号。

  --gpu=<id>                  GPU 列表。[默认: 0]
  --nr_types=<n>              要预测的细胞核类型数量。[默认: 0]
  --type_info_path=<path>     JSON 文件路径，定义类型 ID、类型名称和叠加颜色之间的映射。
                              [默认: '']

  --model_path=<path>         已保存检查点的路径。
  --model_mode=<mode>         原始 HoVer-Net 或用于 PanNuke 和 MoNuSAC 的精简版本，
                              'original' 或 'fast'。[默认: fast]
  --nr_inference_workers=<n>  推理时的 worker 数量。[默认: 8]
  --nr_post_proc_workers=<n>  后处理时的 worker 数量。[默认: 16]
  --batch_size=<n>            每个 GPU 的批量大小。[默认: 32]

两种命令模式 `tile` 和 `wsi`，用于进入相应的推理模式
    tile  对图像块进行推理
    wsi   对全切片图像进行推理

使用 `run_infer.py <命令> --help` 查看各命令的选项和用法。
"""

tile_cli = """
Arguments for processing tiles.

usage:
    tile (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map] [--mem_usage=<n>]
    
options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --mem_usage=<n>        Declare how much memory (physical + swap) should be used for caching. 
                          By default it will load as many tiles as possible till reaching the 
                          declared limit. [default: 0.2]
   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
"""
# tile_cli 中文版
tile_cli_cn = """
处理图像块的参数。

用法:
    tile (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map] [--mem_usage=<n>]
    
选项:
   --input_dir=<path>     输入数据目录路径。假设文件没有嵌套在子目录中。
   --output_dir=<path>    输出目录路径。

   --mem_usage=<n>        指定用于缓存的内存（物理内存 + 交换空间）大小。
                          默认情况下，它会加载尽可能多的图像块，直到达到声明的限制。
                          [默认: 0.2]
   --draw_dot             在叠加图上绘制细胞核质心。[默认: False]
   --save_qupath          可选输出 QuPath v0.2.3 兼容格式。[默认: False]
   --save_raw_map         是否保存原始预测结果。[默认: False]
"""

wsi_cli = """
Arguments for processing wsi

usage:
    wsi (--input_dir=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
        [--cache_path=<path>] [--input_mask_dir=<path>] \
        [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
        [--save_thumb] [--save_mask]
    
options:
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
"""
# wsi_cli 中文版
wsi_cli_cn = """
处理全切片图像的参数

用法:
    wsi (--input_dir=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
        [--cache_path=<path>] [--input_mask_dir=<path>] \
        [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
        [--save_thumb] [--save_mask]
    
选项:
    --input_dir=<path>      输入数据目录路径。假设文件没有嵌套在子目录中。
    --output_dir=<path>     输出目录路径。
    --cache_path=<path>     缓存路径。应放置在至少有 100GB 空间的 SSD 上。[默认: cache]
    --mask_dir=<path>       包含组织掩码的目录路径。
                            应与相应的 WSI 文件同名。[默认: '']

    --proc_mag=<n>          WSI 处理使用的放大倍数（物镜倍率）。[默认: 40]
    --ambiguous_size=<int>  定义沿分块网格的模糊区域，用于执行重后处理。[默认: 128]
    --chunk_shape=<n>       处理时数据块的形状。[默认: 10000]
    --tile_shape=<n>        处理时图像块的形状。[默认: 2048]
    --save_thumb            是否保存缩略图。[默认: False]
    --save_mask             是否保存掩码。[默认: False]
"""

import torch
import logging
import os
import copy
from misc.utils import log_info
from docopt import docopt

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    sub_cli_dict = {'tile' : tile_cli, 'wsi' : wsi_cli}
    args = docopt(__doc__, help=False, options_first=True, 
                    version='HoVer-Net Pytorch Inference v1.0')
    sub_cmd = args.pop('<command>')
    sub_cmd_args = args.pop('<args>')

    # ! TODO: where to save logging
    # ! TODO: 日志保存到哪里
    logging.basicConfig(
        level=logging.INFO,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    if args['--help'] and sub_cmd is not None:
        if sub_cmd in sub_cli_dict: 
            print(sub_cli_dict[sub_cmd])
        else:
            print(__doc__)
        exit()
    if args['--help'] or sub_cmd is None:
        print(__doc__)
        exit()

    sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)
    
    args.pop('--version')
    gpu_list = args.pop('--gpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    nr_gpus = torch.cuda.device_count()
    log_info('Detect #GPUS: %d' % nr_gpus)

    args = {k.replace('--', '') : v for k, v in args.items()}
    sub_args = {k.replace('--', '') : v for k, v in sub_args.items()}
    if args['model_path'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')

    nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
    method_args = {
        'method' : {
            'model_args' : {
                'nr_types'   : nr_types,
                'mode'       : args['model_mode'],
            },
            'model_path' : args['model_path'],
        },
        'type_info_path'  : None if args['type_info_path'] == '' \
                            else args['type_info_path'],
    }

    # ***
    # ***
    run_args = {
        'batch_size' : int(args['batch_size']) * nr_gpus,

        'nr_inference_workers' : int(args['nr_inference_workers']),
        'nr_post_proc_workers' : int(args['nr_post_proc_workers']),
    }

    if args['model_mode'] == 'fast':
        run_args['patch_input_shape'] = 256
        run_args['patch_output_shape'] = 164
    else:
        run_args['patch_input_shape'] = 270
        run_args['patch_output_shape'] = 80

    if sub_cmd == 'tile':
        run_args.update({
            'input_dir'      : sub_args['input_dir'],
            'output_dir'     : sub_args['output_dir'],

            'mem_usage'   : float(sub_args['mem_usage']),
            'draw_dot'    : sub_args['draw_dot'],
            'save_qupath' : sub_args['save_qupath'],
            'save_raw_map': sub_args['save_raw_map'],
        })

    if sub_cmd == 'wsi':
        run_args.update({
            'input_dir'      : sub_args['input_dir'],
            'output_dir'     : sub_args['output_dir'],
            'input_mask_dir' : sub_args['input_mask_dir'],
            'cache_path'     : sub_args['cache_path'],

            'proc_mag'       : int(sub_args['proc_mag']),
            'ambiguous_size' : int(sub_args['ambiguous_size']),
            'chunk_shape'    : int(sub_args['chunk_shape']),
            'tile_shape'     : int(sub_args['tile_shape']),
            'save_thumb'     : sub_args['save_thumb'],
            'save_mask'      : sub_args['save_mask'],
        })
    # ***
    # ***
    
    if sub_cmd == 'tile':
        from infer.tile import InferManager
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)
