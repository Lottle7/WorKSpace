import pyautogui
import time
import threading


def auto_click():
    """自动点击函数"""
    while True:
        try:
            # 获取当前鼠标位置
            current_x, current_y = pyautogui.position()

            # 执行点击
            pyautogui.click()

            print(f"点击执行于位置: ({current_x}, {current_y}) - {time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 等待5分钟
            time.sleep(305)  # 300秒 = 5分钟

        except KeyboardInterrupt:
            print("\n程序已终止")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            time.sleep(305)


if __name__ == "__main__":
    print("自动点击脚本已启动")
    print("点击间隔: 5分钟")
    print("按 Ctrl+C 终止程序")

    # 设置安全特性，鼠标移动到角落时终止
    pyautogui.FAILSAFE = True
    time.sleep(5);
    auto_click()