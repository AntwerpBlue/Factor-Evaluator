# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:46:26 2026

@author: yangxiaozhou
"""

import socket
import json
import os
import polars as pl
import io
import time

STATUS_SUCCESS = 0
STATUS_ERROR = -1


def _log(msg: str):
    print(msg, flush = True)

################## 通过socket接收数据 ################
def _recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError('connection closed unexpectedly')
        data += chunk
    return data


################## 通过socket发送dataframe ################
def sender_to_server(sock, header, df):
    try:
        header_data = json.dumps(header).encode('utf-8')
        sock.sendall(len(header_data).to_bytes(4, byteorder = 'big'))
        sock.sendall(header_data)

        bytes_io_object = io.BytesIO()
        df.write_parquet(file = bytes_io_object)
        parquet_bytes = bytes_io_object.getvalue()
        data_size = len(parquet_bytes)

        sock.sendall(data_size.to_bytes(8, byteorder = 'big'))
        sock.sendall(parquet_bytes)
        bytes_io_object.close()

        _log(f"Sent payload: {data_size / 1024 / 1024:.2f} MB, shape={df.shape}")
        return True
    except Exception as e:
        _log(f"Send to server failed: {e}")
        return False

################### 发送因子至服务器并接收测试结果 ##############
def client_transceiver(
        server_host: str,
        server_port: int,
        df: pl.DataFrame,
        start_date: str,
        end_date: str,
        out_of_sample_start: str,
        save_path: str,
        cluster_name: str = 'default_cluster',
        export_svg: bool = True,
        export_excel: bool = True,
        export_parquet: bool = True,
        k: float = 0.5,
        power: float = 2.0,
        ret_type: str = 'ret',
        single_interval: bool = False,
        test_basic_features: bool = True,
        test_basic_return: bool = True,
        test_style_features: bool = True,
    ):
    ret_type = str(ret_type).strip().lower()
    if ret_type not in {'ret', 'pureret'}:
        raise ValueError(f"Invalid ret_type: {ret_type}. Expected 'ret' or 'pureret'.")
    if cluster_name is None or str(cluster_name).strip() == '':
        cluster_name = 'default_cluster'
    result_obj = {
        'success': False,
        'cluster_name': cluster_name,
        'save_dir': os.path.join(save_path, cluster_name),
        'files': [],
        'result_df': None,
    }

    header = {
        'start_date': start_date,
        'end_date': end_date,
        'out_of_sample_start': out_of_sample_start,
        'cluster_name': cluster_name,
        'export_svg': bool(export_svg),
        'export_excel': bool(export_excel),
        'export_parquet': bool(export_parquet),
        'k': float(k),
        'power': float(power),
        'ret_type': ret_type,
        'single_interval': bool(single_interval),
        'test_basic_features': bool(test_basic_features),
        'test_basic_return': bool(test_basic_return),
        'test_style_features': bool(test_style_features),
    }

    save_dir = result_obj['save_dir']
    os.makedirs(save_dir, exist_ok = True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _log(f"Connecting to server {server_host}:{server_port} ...")
        sock.settimeout(3000)
        sock.connect((server_host, server_port))
        _log('Connected.')
    except Exception as e:
        _log(f"Connect failed: {e}")
        return result_obj

    try:
        success_flag = sender_to_server(sock, header, df)
        if not success_flag:
            raise RuntimeError('send request failed')

        _log('Waiting for server response ...')
        status_data = _recv_exact(sock, 1)
        status_code = int.from_bytes(status_data, byteorder = 'big', signed = True)

        if status_code == STATUS_SUCCESS:
            n_files_data = _recv_exact(sock, 4)
            n_files = int.from_bytes(n_files_data, byteorder = 'big')
            _log(f"Receiving {n_files} output files ...")

            for i in range(n_files):
                filename_len_data = _recv_exact(sock, 4)
                filename_len = int.from_bytes(filename_len_data, byteorder = 'big')
                filename_data = _recv_exact(sock, filename_len)
                output_filename = filename_data.decode('utf-8')

                file_size_data = _recv_exact(sock, 8)
                file_size = int.from_bytes(file_size_data, byteorder = 'big')
                file_data = _recv_exact(sock, file_size)
                if output_filename == 'result_all.parquet':
                    result_obj['result_df'] = pl.read_parquet(io.BytesIO(file_data))
                    result_obj['files'].append(output_filename)
                    _log(f"[{i+1}/{n_files}] received in-memory {output_filename}")
                    continue

                file_path = os.path.join(save_dir, output_filename)
                with open(file_path, 'wb') as f:
                    f.write(file_data)

                result_obj['files'].append(output_filename)
                _log(f"[{i+1}/{n_files}] saved {file_path}")

            result_obj['success'] = True
            return result_obj

        error_len_data = _recv_exact(sock, 4)
        error_len = int.from_bytes(error_len_data, byteorder = 'big')
        error_data = _recv_exact(sock, error_len)
        error_msg = error_data.decode('utf-8')
        _log(f"Server returned error: {error_msg}")
        return result_obj

    except socket.timeout:
        _log('Connection timeout while waiting for response.')
        return result_obj
    except Exception as e:
        _log(f"Client error: {e}")
        return result_obj
    finally:
        sock.close()

def main():
    t0 = time.time()
    _log('Client started.')
    SERVER_HOST = '192.168.1.97' # 测试服务器配置在meta1上
    SERVER_PORT = 5000
    
    ############# 参数配置 ###############
    start_date = '20160101'
    out_of_sample_start = '20250101'
    end_date = '20260204'
    EXPORT_SVG = True
    EXPORT_EXCEL = True
    EXPORT_PARQUET = True
    SINGLE_INTERVAL = False
    TEST_BASIC_FEATURES = True
    TEST_BASIC_RETURN = True
    TEST_STYLE_FEATURES = True
    TOPK_K = 1.0
    TOPK_POWER = 2.0
    RET_TYPE = 'pureret'
    SINGLE_INTERVAL = False
    TEST_BASIC_FEATURES = True
    TEST_BASIC_RETURN = True
    TEST_STYLE_FEATURES = True
    save_path = 'report'
    os.makedirs(save_path, exist_ok = True)
    ############# 参数配置 ###############
        
    ############ 读取因子或因子簇 ###########
    """
    因子文件格式（polars dataframe,长表）：
    - 日期列（名为TRADE_DT或date，格式为YYYYMMDD的str）
    - 股票代码列（名为WINDCODE或header，格式如000001.SZ的9位字符串
    - 每一个因子为一列，T日的因子值需T日收盘可交易（一般为T-1日盘后生成）
    """
    df = ... # 因子polars dataframe
    cluster_name = ...   # 因子簇名称
    
    ############ 发送因子进行测试并保存测试结果 ##############
    result = client_transceiver(
        SERVER_HOST,
        SERVER_PORT,
        df,
        start_date,
        end_date,
        out_of_sample_start,
        save_path,
        cluster_name,
        export_svg = EXPORT_SVG,
        export_excel = EXPORT_EXCEL,
        export_parquet = EXPORT_PARQUET,
        k = TOPK_K,
        power = TOPK_POWER,
        ret_type = RET_TYPE,
        single_interval = SINGLE_INTERVAL,
        test_basic_features = TEST_BASIC_FEATURES,
        test_basic_return = TEST_BASIC_RETURN,
        test_style_features = TEST_STYLE_FEATURES,
    )
    if result['success']:
        _log('Backtest finished.')
        _log(f"Saved folder: {result['save_dir']}")
        _log(f"Files: {result['files']}")
        if result['result_df'] is not None:
            _log(f"result_df shape: {result['result_df'].shape}")
    else:
        _log('Backtest failed.')
   

if __name__ == '__main__':
    main()
