import cv2

# Test the ports and returns a tuple with the available ports and the ones that are working.
def list_ports():
    max_num_webcams = 5
    working_ports = []
    available_ports = []
    for dev_port in range(max_num_webcams):

        try: 
            camera = cv2.VideoCapture(dev_port, cv2.CAP_DSHOW)  #for USB webcam
        except:
            break

        if not camera.isOpened():
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
    return available_ports,working_ports

if __name__ == "__main__":
    list_ports()