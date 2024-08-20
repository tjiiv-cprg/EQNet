import sys, os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit, QSizePolicy, QMessageBox, QShortcut, QMenuBar, QMenu
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QBrush, QColor, QFont
from PyQt5.Qt import Qt, QKeySequence

class ImageViewer(QMainWindow):
    def __init__(self, folder1, folder2, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.folder1 = folder1
        self.folder2 = folder2
        self.image1_paths = []
        self.image2_paths = []
        self.current_index = 0
        self.initUI()
        self.scan_folders()

    def initUI(self):
        # Create main layout
        self.central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Create image display layout
        self.image1_layout = QVBoxLayout()
        self.image1_label = QLabel()
        self.image1_label.setScaledContents(True)  # Enable image scaling
        self.image1_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy for image1_label
        self.image1_path_label = QLabel()
        self.image1_layout.addWidget(self.image1_label)
        self.image1_layout.addWidget(self.image1_path_label)

        self.image2_layout = QVBoxLayout()
        self.image2_label = QLabel()
        self.image2_label.setScaledContents(True)  # Enable image scaling
        self.image2_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy for image2_label
        self.image2_path_label = QLabel()
        self.image2_layout.addWidget(self.image2_label)
        self.image2_layout.addWidget(self.image2_path_label)

        image_layout = QHBoxLayout()
        image_layout.addLayout(self.image1_layout)
        image_layout.addLayout(self.image2_layout)

        # Create navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.addStretch(10)
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_pair)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_pair)
        self.jump_label = QLabel("Jump to:")
        self.jump_input = QLineEdit()
        self.jump_input.returnPressed.connect(self.jump_to_pair)
        self.jump_input.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.jump_label)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.jump_input)
        nav_layout.addWidget(self.next_button)
        nav_layout.addStretch(3)

        # Add layouts to main layout
        main_layout.addLayout(image_layout)
        main_layout.addLayout(nav_layout)

        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle("Parallel Viewer")

        # Set the window size policy to allow resizing
        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.setMinimumSize(600, 400)

        # Create the icon
        icon_size = 64
        icon_pixmap = QPixmap(icon_size, icon_size)
        icon_pixmap.fill(Qt.transparent)
        painter = QPainter(icon_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(76, 175, 80)))
        painter.drawRoundedRect(0, 0, icon_size, icon_size, 8, 8)
        painter.setFont(QFont("Arial", 32, QFont.Bold))
        painter.setPen(Qt.white)
        painter.drawText(0, 0, icon_size, icon_size, Qt.AlignCenter, "P")
        painter.end()

        # Set the window icon
        self.setWindowIcon(QIcon(icon_pixmap))

        # Create the menu bar
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        # Create the "Help" menu
        self.help_menu = self.menu_bar.addMenu("Help")
        self.help_action = self.help_menu.addAction("How to Use")
        self.help_action.triggered.connect(self.show_help)

        # Add event handlers for image path labels
        self.image1_path_label.mousePressEvent = self.copy_image1_path
        self.image1_path_label.mouseDoubleClickEvent = self.copy_both_paths
        self.image2_path_label.mousePressEvent = self.copy_image2_path
        self.image2_path_label.mouseDoubleClickEvent = self.copy_both_paths

        # Add a shortcut to copy both paths
        self.copy_both_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_C), self)
        self.copy_both_shortcut.activated.connect(self.copy_both_paths)

    def copy_image1_path(self, event):
        QApplication.clipboard().setText(self.image1_paths[self.current_index])

    def copy_image2_path(self, event):
        QApplication.clipboard().setText(self.image2_paths[self.current_index])

    def copy_both_paths(self, event):
        QApplication.clipboard().setText(f"{self.image1_paths[self.current_index]}\n{self.image2_paths[self.current_index]}")

    def show_image_pair(self, index):
        try:
            image1_path = self.image1_paths[index]
            image2_path = self.image2_paths[index]
            image1_pixmap = QPixmap(image1_path)
            image2_pixmap = QPixmap(image2_path)
            self.image1_label.setPixmap(image1_pixmap.scaled(self.image1_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # self.image1_label.setPixmap(image1_pixmap)
            self.image2_label.setPixmap(image2_pixmap.scaled(self.image2_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # self.image2_label.setPixmap(image2_pixmap)
            self.image1_path_label.setText(image1_path)
            self.image2_path_label.setText(image2_path)
            self.jump_input.setText(str(self.current_index + 1))  # Update the text box with the current index
        except IndexError:
            print(f"Error: Index {index} is out of range for the provided image pairs.")
        except FileNotFoundError:
            print(f"Error: One or more image files not found at the provided paths.")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_image_pair(self.current_index)

    def show_previous_pair(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image_pair(self.current_index)

    def show_next_pair(self):
        if self.current_index < len(self.image1_paths) - 1:
            self.current_index += 1
            self.show_image_pair(self.current_index)

    def jump_to_pair(self):
        try:
            index = int(self.jump_input.text()) - 1
            if 0 <= index < len(self.image1_paths):
                self.current_index = index
                self.show_image_pair(self.current_index)
        except ValueError:
            print("Error: Invalid input. Please enter a valid integer.")
        self.jump_input.clear()

    def scan_folders(self):
        # Scan folder1 and get the image paths
        for filename in sorted(os.listdir(self.folder1)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                self.image1_paths.append(os.path.join(self.folder1, filename))

        # Scan folder2 and get the image paths
        for filename in sorted(os.listdir(self.folder2)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                self.image2_paths.append(os.path.join(self.folder2, filename))

        # Ensure that the number of images in both folders are equal
        if len(self.image1_paths) != len(self.image2_paths):
            QMessageBox.warning(self, "Warning", "The number of images in the two folders is not equal.")
            return

        self.show_image_pair(0)
    
    def show_help(self):
        help_text = """
        Parallel Viewer Usage:

        - Click on an image path label to copy the path to the clipboard.
        - Double-click on an image path label to copy both paths to the clipboard.
        - Use the "Previous" and "Next" buttons to navigate through the image pairs.
        - Enter a number in the "Jump to:" box and press Enter to jump to a specific image pair.
        - Use the Ctrl+C shortcut to copy both paths of the current image pair to the clipboard.
        """
        QMessageBox.information(self, "Help", help_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    image_pairs = [
        ("/home/hyuyao/2024/trackpred_v2/results/PLOTS/EQNet-Viz/00000.png", "/home/hyuyao/2024/trackpred_v2/results/PLOTS/EQNet-Viz/00001.png"),
        ("/home/hyuyao/2024/trackpred_v2/results/PLOTS/EQNet-Viz/00002.png", "/home/hyuyao/2024/trackpred_v2/results/PLOTS/EQNet-Viz/00003.png"),
        ("/home/hyuyao/2024/trackpred_v2/results/PLOTS/EQNet-Viz/00004.png", "/home/hyuyao/2024/trackpred_v2/results/PLOTS/EQNet-Viz/00005.png"),
    ]

    viewer = ImageViewer(
        folder1="results/PLOTS/EQNet-Viz/",
        folder2="results/PLOTS/QCNet-Viz/"
    )
    viewer.show()
    sys.exit(app.exec_())