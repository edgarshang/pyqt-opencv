import sys
from PyQt5.QtWidgets import QApplication

from UIQt import UI_Image
from imageProcess import opencvImage
from yolov5_test import YOLOV5_Process

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
    
    m_ui = UI_Image()
    m_ui.setImagePorcess(imageProc)
    m_ui.setYoloV5Process(yolov5)

    yolov5.setShowImage(m_ui.showMatImage)

    m_ui.show()
    sys.exit(app.exec_())
