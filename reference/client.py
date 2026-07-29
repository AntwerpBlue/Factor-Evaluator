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


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError('connection closed unexpectedly')
        data += chunk
    return data


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
        extra_delay_day: bool = True,
    ):
    ret_type = str(ret_type).strip().lower()
    if ret_type not in {'ret', 'barraret'}:
        raise ValueError(f"Invalid ret_type: {ret_type}. Expected 'ret' or 'barraret'.")
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
        'extra_delay_day': bool(extra_delay_day),
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


def load_factor_cluster_df(factor_folder: str, max_files = None) -> pl.DataFrame:
    t_list = time.time()
    factor_files = [
        os.path.join(factor_folder, x)
        for x in os.listdir(factor_folder)
        if x.lower().endswith('.parquet')
    ]
    factor_files = sorted(factor_files)
    _log(f'Found {len(factor_files)} parquet files. list cost={time.time() - t_list:.2f}s')
    if not factor_files:
        raise FileNotFoundError(f'No parquet files found: {factor_folder}')
    if max_files is not None:
        factor_files = factor_files[:max_files]
        _log(f'Using first {len(factor_files)} files for debug run.')
    t_load = time.time()
    _log('Loading factor parquet files ...')
    df = (
        pl.concat(
            [pl.scan_parquet(x) for x in factor_files],
            how = 'vertical'
        )
        .collect()
    )
    if 'trade_dt' in df.columns:
        try:
            df = df.with_columns(pl.col('trade_dt').dt.strftime('%Y%m%d').alias('trade_dt'))
        except Exception:
            df = df.with_columns(pl.col('trade_dt').cast(pl.Utf8).alias('trade_dt'))
    rename_map = {}
    if 'trade_dt' in df.columns:
        rename_map['trade_dt'] = 'date'
    elif 'TRADE_DT' in df.columns:
        rename_map['TRADE_DT'] = 'date'
    if 'code' in df.columns:
        rename_map['code'] = 'header'
    elif 'WINDCODE' in df.columns:
        rename_map['WINDCODE'] = 'header'
    if rename_map:
        df = df.rename(rename_map)
    _log(f'Factor dataframe ready. shape={df.shape}, load cost={time.time() - t_load:.2f}s')
    return df


def main():
    t0 = time.time()
    _log('Client started.')
    SERVER_HOST = '127.0.0.1'
    SERVER_PORT = 5000
    factor_root = r'\\dbserverRS\Intern\LikeruiShare\Factor_组合'
   # cluster_list = ['Factor_TreatEqually','Factor_Water','Factor_tide_complete','Factor_SurgingStream']
   # cluster_list = ["Factor_Hide",'Factor_LGBM','Factor_climb_peak','Factor_Water','Factor_STR_STT_STT2_STV']
  #  cluster_list = ["Factor_L1D5M0","Factor_L1D5M0R","Factor_L2D5M0","Factor_L2D5M0R","Factor_L5D5M0","Factor_L5D5M0R",]
    # cluster_list = ["Factor_Bull_Bear_Game","Factor_climb_peak","Factor_cloud_clear_fog_disperse","Factor_Dripstone","Factor_hf_amplitude",
    #                 "Factor_Hide","Factor_LGBM","Factor_moderate_risk","Factor_moth_to_flame","Factor_PeakRidgeValley","Factor_Sheep",
    #                 "Factor_STR_STT_STT2_STV", "Factor_SurgingStream", "Factor_team_coin","Factor_tide_complete","Factor_TreatEqually",
    #                 "Factor_Wait","Factor_Water",
    #                 ]   
    cluster_list = ['Factor_densemble_L2D5M0']
    # Debug option: set MAX_FILES_PER_CLUSTER to a small number (e.g. 50) for quick smoke runs.
    MAX_FILES_PER_CLUSTER = None

    start_date = '20220101'
    out_of_sample_start = '20250101'
    end_date = '20260204'
    EXPORT_SVG = True
    EXPORT_EXCEL = True
    EXPORT_PARQUET = True
    TOPK_K = 1.0
    TOPK_POWER = 2.0
    RET_TYPE = 'barraret'
    EXTRA_DELAY_DAY = True
    save_path = 'report'
    os.makedirs(save_path, exist_ok = True)
    all_results = {}
    for idx, cluster_name in enumerate(cluster_list):
        _log(f"\n===== [{idx+1}/{len(cluster_list)}] Cluster: {cluster_name} =====")
        factor_folder = os.path.join(factor_root, cluster_name)
        _log(f'Factor folder: {factor_folder}')
        if not os.path.exists(factor_folder):
            _log(f'Cluster folder not found, skip: {factor_folder}')
            all_results[cluster_name] = {
                'success': False,
                'cluster_name': cluster_name,
                'save_dir': os.path.join(save_path, cluster_name),
                'files': [],
                'result_df': None,
                'error': 'cluster folder not found',
            }
            continue
        try:
            df = load_factor_cluster_df(factor_folder, max_files = MAX_FILES_PER_CLUSTER)
        except Exception as e:
            _log(f'Load cluster failed: {e}')
            all_results[cluster_name] = {
                'success': False,
                'cluster_name': cluster_name,
                'save_dir': os.path.join(save_path, cluster_name),
                'files': [],
                'result_df': None,
                'error': str(e),
            }
            continue
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
            extra_delay_day = EXTRA_DELAY_DAY,
        )
        all_results[cluster_name] = result
        if result['success']:
            _log('Backtest finished.')
            _log(f"Saved folder: {result['save_dir']}")
            _log(f"Files: {result['files']}")
            if result['result_df'] is not None:
                _log(f"result_df shape: {result['result_df'].shape}")
        else:
            _log('Backtest failed.')
    ok_clusters = [k for k, v in all_results.items() if v.get('success')]
    fail_clusters = [k for k, v in all_results.items() if not v.get('success')]
    _log(f'Cluster summary: success={len(ok_clusters)}, failed={len(fail_clusters)}')
    if fail_clusters:
        _log(f'Failed clusters: {fail_clusters}')
    _log(f'Total client elapsed: {time.time() - t0:.2f}s')


if __name__ == '__main__':
    main()
