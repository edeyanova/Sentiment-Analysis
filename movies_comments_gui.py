import sys
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit,
                             QTextEdit, QGridLayout, QApplication,
                             QPushButton)
import SentimentAnalysisModule.sentiment_analysis_module as s

class MoviesCommentsWindow(QWidget):
    
    
    
    def __init__(self):
        super().__init__()

        self.comments = QLabel('Comments')
        self.commentInput = QTextEdit()
        self.classifyButton = QPushButton('Classify')
        self.commentClass = QLabel('Class')
        self.classOutput = QLineEdit()
        

        self.initUI()

    def initUI(self):      
        self.classOutput.setReadOnly(True)
        

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.comments, 1, 0)
        grid.addWidget(self.commentInput, 1, 1, 5, 2)
        grid.addWidget(self.classifyButton, 1, 3)
        grid.addWidget(self.commentClass, 6, 0)
        grid.addWidget(self.classOutput, 6, 1)
        

        self.classifyButton.clicked.connect(self.classify)

        self.setLayout(grid)
        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('Review')
        self.show()
        
    def classify(self):
        comment = self.commentInput.toPlainText()
        if comment is not "":
            classification = s.classify(comment)
            commentClass = ''
            if classification == 'Pos':
                commentClass = 'positive'
            else:
                commentClass = 'negative'
            
        
            self.classOutput.setText(commentClass)
            
        
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MoviesCommentsWindow()
    sys.exit(app.exec_())

