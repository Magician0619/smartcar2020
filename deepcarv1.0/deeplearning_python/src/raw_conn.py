from ctypes import cdll
import sys
sys.stdout.write("Loading Driver...")
so = cdll.LoadLibrary
driver = so("/home/root/workspace/deepcar/deeplearning_python/lib/libart_driver.so")
driver.art_racecar_init(38400, "/dev/ttyUSB0".encode("utf-8"))
print("Done.")
while True:
    try:
        ps = input("Input Params:")
        p1,p2 = ps.strip().split(" ")
        p1 = int(p1)
        p2 = int(p2)
        driver.send_cmd(p1,p2)
        print(p1,p2,"sent.")
    except Exception as e:
        sys.stderr.write("Error:"+str(e)+"\n")
