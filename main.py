import sys
from PyQt5.QtWidgets import QApplication

from UIQt import UI_Image
from imageProcess import opencvImage
from yolov5_test import YOLOV5_Process
from yolov8_test import YOLOV8_Process
from landmark.landmark_test import landMark_Process
from age_gentle.age_gentle_test import AgeGentle_Process

import platform
import os
import PySide2

if platform.system() == "Windows":
    pass
elif platform.system() == "Linux":
    dirname = os.path.dirname(PySide2.__file__)
    plugin_path = os.path.join(dirname, "plugins", "platforms")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
else:
    pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    imageProc = opencvImage()
    yolov5 = YOLOV5_Process()
    yolov8 = YOLOV8_Process()
    landmark = landMark_Process()
    ageGentle = AgeGentle_Process()
    
    m_ui = UI_Image()
    m_ui.setImagePorcess(imageProc)
    m_ui.setYoloV5Process(yolov5)
    m_ui.setYoloV8Process(yolov8)
    m_ui.setLandMarkProcess(landmark)
    m_ui.setAgeGentleProcess(ageGentle)

    yolov5.setShowImage(m_ui.showMatImage)
    yolov8.setImageShower(m_ui)
    landmark.setImageShower(m_ui)
    ageGentle.setImageShower(m_ui)

    m_ui.show()
    sys.exit(app.exec_())
