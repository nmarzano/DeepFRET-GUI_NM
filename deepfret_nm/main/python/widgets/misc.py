import multiprocessing

from main.python.widgets.inspectors import SheetInspector

multiprocessing.freeze_support()

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class ExportDialog(QFileDialog):
    """
    Custom export dialog to change labels on the accept button.
    Cancel button doesn't work for whatever reason on MacOS (Qt bug?).
    """

    def __init__(self, init_dir, accept_label="Accept"):
        super().__init__()
        self.setFileMode(self.DirectoryOnly)
        self.setLabelText(self.Accept, accept_label)
        self.setDirectory(init_dir)


class ProgressBar(QProgressDialog, SheetInspector):
    """
    Displays a progressbar, using known length of a loop.
    """

    def __init__(self, parent, loop_len=0):
        super().__init__(parent=parent)
        self.minimumSizeHint()
        self.setValue(0)
        self.setMinimum(0)
        self.setMaximum(
            loop_len
        )  # Corrected because iterations start from zero, but minimum length is 1
        self.show()

    def increment(self):
        """
        Increments progress by 1
        """
        self.setValue(self.value() + 1)


class UpdatingList:
    """
    Class for dynamically updating attributes that need to be iterable,
    e.g. instead of keeping separate has_blu, has_grn, has_red checks,
    collect them in this class to make them mutable.
    """

    def __iter__(self):
        return (
            self.__getattribute__(i)
            for i in dir(self)
            if not i.startswith("__")
        )


class CheckBoxDelegate(QStyledItemDelegate):
    """
    Implement into dynamic checkboxes to check their states (see also ListView below).
    """

    def editorEvent(self, event, model, option, index):
        checked = index.data(Qt.CheckStateRole)
        ret = QStyledItemDelegate.editorEvent(self, event, model, option, index)

        if checked != index.data(Qt.CheckStateRole):
            self.parent().checked.emit(index)
        return ret


class Delegate(QStyledItemDelegate):
    """
    Triggers a return event whenever a checkbox is triggered.
    """

    def editorEvent(self, event, model, option, index):
        checked = index.data(Qt.CheckStateRole)
        ret = QStyledItemDelegate.editorEvent(self, event, model, option, index)
        if checked != index.data(Qt.CheckStateRole):
            self.parent().checked.emit(index)
        return ret


class ListView(QListView):
    """
    Custom ListView implementation which handles checkbox triggers.
    """

    checked = pyqtSignal(QModelIndex)

    def __init__(self, *args, **kwargs):
        super(ListView, self).__init__(*args, **kwargs)
        self.setItemDelegate(Delegate(self))
        self.setMaximumWidth(450)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
