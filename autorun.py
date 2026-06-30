import os
import sys
import time
import subprocess
from datetime import datetime


# ==================== 配置区域 ====================1

SCRIPTS = [
    r'C:\Users\parag\OneDrive\Desktop\Documents\ML\online data enhanced\EfficientNet-B0.py',
    r'C:\Users\parag\OneDrive\Desktop\Documents\ML\online data enhanced\MobileNetv3_Large.py',
    r'C:\Users\parag\OneDrive\Desktop\Documents\ML\online data enhanced\ResNet18.py',
    r'C:\Users\parag\OneDrive\Desktop\Documents\ML\online data enhanced\ResNet34.py',
    r'C:\Users\parag\OneDrive\Desktop\Documents\ML\online data enhanced\ResNet50.py',
    r'C:\Users\parag\OneDrive\Desktop\Documents\ML\online data enhanced\DenseNet121.py'
]

# 每个脚本运行次数
RUN_COUNT = 3

# 使用当前运行 autorun.py 的 Python
PYTHON_EXE = sys.executable

# autorun.py 所在文件夹
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 本次自动运行的时间戳
SESSION_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')

# 日志文件夹
LOG_DIR = os.path.join(BASE_DIR, f'autorun_logs_{SESSION_TIME}')

# 主日志文件
MAIN_LOG_FILE = os.path.join(LOG_DIR, 'auto_run_log.txt')

# 每个脚本每次运行之间休息秒数
SLEEP_SECONDS = 2

# ==================================================


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def now_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_message(message):
    ensure_log_dir()

    log_line = f'[{now_time()}] {message}'
    print(log_line, flush=True)

    with open(MAIN_LOG_FILE, 'a', encoding='utf-8') as file:
        file.write(log_line + '\n')


def safe_filename(name):
    return (
        name.replace(':', '_')
            .replace('\\', '_')
            .replace('/', '_')
            .replace(' ', '_')
    )


def check_all_scripts():
    log_message('开始检查脚本路径是否存在...')

    all_ok = True

    for index, script_path in enumerate(SCRIPTS, start=1):
        if os.path.exists(script_path):
            log_message(f'✅ 脚本 {index} 存在: {script_path}')
        else:
            log_message(f'❌ 脚本 {index} 不存在: {script_path}')
            all_ok = False

    return all_ok


def run_script(script_path, run_number):
    script_name = os.path.basename(script_path)
    script_dir = os.path.dirname(script_path)

    log_message('')
    log_message('-' * 80)
    log_message(f'开始运行: {script_name}')
    log_message(f'运行次数: 第 {run_number}/{RUN_COUNT} 次')
    log_message(f'脚本目录: {script_dir}')
    log_message('-' * 80)

    if not os.path.exists(script_path):
        log_message(f'❌ 文件不存在: {script_path}')
        return False

    if not os.path.isdir(script_dir):
        log_message(f'❌ 脚本所在文件夹不存在: {script_dir}')
        return False

    run_log_name = f'{safe_filename(script_name)}_run_{run_number}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    run_log_path = os.path.join(LOG_DIR, run_log_name)

    start_time = time.time()

    try:
        with open(run_log_path, 'w', encoding='utf-8', errors='replace') as run_log:
            run_log.write(f'脚本: {script_path}\n')
            run_log.write(f'运行次数: {run_number}/{RUN_COUNT}\n')
            run_log.write(f'开始时间: {now_time()}\n')
            run_log.write(f'Python: {PYTHON_EXE}\n')
            run_log.write(f'工作目录: {script_dir}\n')
            run_log.write('=' * 80 + '\n\n')
            run_log.flush()

            process = subprocess.Popen(
                [PYTHON_EXE, script_path],
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )

            for line in process.stdout:
                print(line, end='', flush=True)
                run_log.write(line)
                run_log.flush()

            return_code = process.wait()

        duration = time.time() - start_time

        if return_code == 0:
            log_message(f'✅ 完成: {script_name},第 {run_number} 次')
            log_message(f'耗时: {duration / 60:.2f} 分钟')
            log_message(f'详细日志: {run_log_path}')
            return True
        else:
            log_message(f'❌ 失败: {script_name},第 {run_number} 次')
            log_message(f'返回码: {return_code}')
            log_message(f'请查看详细日志: {run_log_path}')
            return False

    except Exception as error:
        duration = time.time() - start_time
        log_message(f'❌ 运行出错: {script_name},第 {run_number} 次')
        log_message(f'耗时: {duration / 60:.2f} 分钟')
        log_message(f'错误信息: {error}')
        return False


def main():
    ensure_log_dir()

    log_message('=' * 80)
    log_message('自动运行脚本启动')
    log_message(f'当前 Python: {PYTHON_EXE}')
    log_message(f'日志文件夹: {LOG_DIR}')
    log_message(f'脚本数量: {len(SCRIPTS)}')
    log_message(f'每个脚本运行次数: {RUN_COUNT}')
    log_message(f'总任务数: {len(SCRIPTS) * RUN_COUNT}')
    log_message('=' * 80)

    all_scripts_ok = check_all_scripts()

    if not all_scripts_ok:
        log_message('')
        log_message('❌ 有脚本路径不存在,请先修改路径。程序即将停止。')
        input('\n有脚本路径错误,按回车键退出...')
        return

    total_success = 0
    total_failed = 0
    overall_start = time.time()

    for script_index, script_path in enumerate(SCRIPTS, start=1):
        script_name = os.path.basename(script_path)

        log_message('')
        log_message('*' * 80)
        log_message(f'开始处理脚本 {script_index}/{len(SCRIPTS)}')
        log_message(f'脚本名称: {script_name}')
        log_message('*' * 80)

        script_success = 0
        script_failed = 0

        for run_number in range(1, RUN_COUNT + 1):
            success = run_script(script_path, run_number)

            if success:
                script_success += 1
                total_success += 1
            else:
                script_failed += 1
                total_failed += 1

            if run_number < RUN_COUNT:
                log_message(f'等待 {SLEEP_SECONDS} 秒后继续下一次运行...')
                time.sleep(SLEEP_SECONDS)

        log_message('')
        log_message(f'{script_name} 运行结束')
        log_message(f'成功: {script_success} 次')
        log_message(f'失败: {script_failed} 次')

    total_duration = time.time() - overall_start

    log_message('')
    log_message('=' * 80)
    log_message('所有任务完成')
    log_message(f'总耗时: {total_duration / 60:.2f} 分钟')
    log_message(f'成功任务数: {total_success}')
    log_message(f'失败任务数: {total_failed}')
    log_message(f'主日志文件: {MAIN_LOG_FILE}')
    log_message(f'详细日志文件夹: {LOG_DIR}')
    log_message('=' * 80)

    input('\n全部运行结束,按回车键退出...')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log_message('')
        log_message('⚠️ 用户手动中断运行')
        input('\n程序已被你手动中断,按回车键退出...')
    except Exception as error:
        log_message('')
        log_message(f'❌ 程序发生严重错误: {error}')
        input('\n程序出错,按回车键退出...')