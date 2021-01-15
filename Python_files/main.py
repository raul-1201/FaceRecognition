from Python_files.ui_Project import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.setupUi()  # Setting up the elements in the widget
    ui.show()  # !!!!! Windows are hidden by default.

    # Start the event loop.
    sys.exit(app.exec_())
    # The application will run until I exit and the event loop has stopped.
