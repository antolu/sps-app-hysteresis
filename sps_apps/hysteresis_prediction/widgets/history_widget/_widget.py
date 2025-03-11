from __future__ import annotations

import logging
import typing

from accwidgets import lsa_selector
from op_app_context import context
from qtpy import QtCore, QtGui, QtWidgets

from ...generated.reference_selector_dialog_ui import Ui_ReferenceSelectorDialog
from ...history import PredictionHistory
from ._view import HistoryPlotWidget

log = logging.getLogger(__name__)


class HistoryWidget(QtWidgets.QWidget):
    def __init__(
        self,
        history: PredictionHistory,
        parent: QtWidgets.QWidget | None = None,
        *,
        max_tabs: int = 10,
        measured_available: bool = False,
    ) -> None:
        super().__init__(parent=parent)

        self.measured_available = measured_available
        self.max_tabs = max_tabs
        self.LsaSelector = self._setup_lsa_selector()

        self.listView = QtWidgets.QListView(self)
        self.setReferenceButton = QtWidgets.QPushButton("Set Reference", self)

        self.actionRefreshLsaSelector = QtWidgets.QAction(self)
        self.actionRefreshLsaSelector.setText("&Refresh LSA Selector")
        self.actionRefreshLsaSelector.triggered.connect(self.LsaSelector.model.refetch)

        self.actionImport_Predictions = QtWidgets.QAction(self)
        self.actionImport_Predictions.setText("&Import Predictions")
        self.actionExport_Predictions = QtWidgets.QAction(self)
        self.actionExport_Predictions.setText("&Export Predictions")
        self.actionExit = QtWidgets.QAction(self)
        self.actionExit.setText("&Exit")
        self.actionExit.triggered.connect(self.close)

        self.menubar = QtWidgets.QMenuBar(self)
        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(self.actionRefreshLsaSelector)
        file_menu.addAction(self.actionImport_Predictions)
        file_menu.addAction(self.actionExport_Predictions)
        file_menu.addSeparator()
        file_menu.addAction(self.actionExit)

        view_menu = self.menubar.addMenu("&View")
        self.actionResetAxes = QtWidgets.QAction(self)
        self.actionResetAxes.setText("&Reset Axes")
        view_menu.addAction(self.actionResetAxes)

        # add left widget
        frame = QtWidgets.QFrame(self)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.addWidget(self.LsaSelector)
        layout.addWidget(self.listView)
        layout.addWidget(self.setReferenceButton)
        frame.setMaximumWidth(320)
        frame.setMinimumWidth(320)

        # make the widget
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(frame)
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.layout().addWidget(self.tabWidget)

        # add menubar
        self.layout().setMenuBar(self.menubar)

        self._history = history
        self._tabs: dict[str, HistoryPlotWidget] = {}

        # make tabs closable
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.tabCloseRequested.connect(self.onTabCloseRequested)

        # set minimum size
        if measured_available:
            self.tabWidget.setMinimumSize(1000, 600)
        else:
            self.tabWidget.setMinimumSize(600, 600)

        # set floating
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle("SPS Hysteresis Prediction History")

        self.listView.clicked.connect(self.onItemClicked)

        self.setReferenceButton.setEnabled(False)
        self.setReferenceButton.clicked.connect(self.onSelectReferenceClicked)

    def _setup_lsa_selector(self) -> lsa_selector.LsaSelector:
        selector_model = lsa_selector.LsaSelectorModel(
            accelerator=lsa_selector.LsaSelectorAccelerator.SPS,
            lsa=context.lsa_client,
            categories={
                lsa_selector.AbstractLsaSelectorContext.Category.MD,
                lsa_selector.AbstractLsaSelectorContext.Category.OPERATIONAL,
            },
        )
        LsaSelector = lsa_selector.LsaSelector(model=selector_model, parent=self)
        LsaSelector.contextSelectionChanged.connect(self.onUserChanged)
        LsaSelector.setMaximumWidth(300)
        LsaSelector.setMinimumWidth(300)

        return LsaSelector

    @QtCore.Slot(lsa_selector.AbstractLsaSelectorContext)
    def onUserChanged(self, user: lsa_selector.AbstractLsaSelectorContext) -> None:
        # if button state is off, trigger the button
        log.debug(f"Requested to show {user.name}")
        self.show_or_create_tab(user.name)

    @QtCore.Slot()
    def onSelectReferenceClicked(self) -> None:
        """
        Select a new reference cycle for plots, use a QIdentityProxyModel,
        but select only one.
        """
        dialog = ReferenceSelectorDialog(model=self.currentWidget.lmodel, parent=self)

        def onDialogAcccepted() -> None:
            selected_index = dialog.selected_item
            if selected_index is not None:
                item = self.lmodel.itemAt(selected_index)

                self.currentWidget.lmodel.setReference(item)
                self.currentWidget.pmodel.setReference(item)

        dialog.accepted.connect(onDialogAcccepted)

        dialog.open()

    @QtCore.Slot()
    def onResetAxes(self) -> None:
        if self.tabWidget.currentWidget() is not None:
            typing.cast(HistoryPlotWidget, self.tabWidget.currentWidget()).resetAxes()

    @property
    def currentWidget(self) -> HistoryPlotWidget:
        current_widget = self.tabWidget.currentWidget()
        if current_widget is None:
            msg = "No tab selected."
            raise RuntimeError(msg)

        return typing.cast(HistoryPlotWidget, current_widget)

    @QtCore.Slot(QtCore.QModelIndex)
    def onItemClicked(self, index: QtCore.QModelIndex) -> None:
        self.currentWidget.itemClicked(index)

    def show_or_create_tab(self, name: str) -> None:
        # check if tab is already open and selected
        if name in self._tabs and self.tabWidget.currentWidget() == self._tabs[name]:
            log.debug(f"Tab {name} already open and selected.")
            return

        if name not in self._tabs:
            msg = f"Creating new tab for {name}"
            log.debug(msg)
            widget = HistoryPlotWidget(
                self._history.model(name),
                self,
                plot_measured=self.measured_available,
            )
            self.tabWidget.addTab(widget, name)
            self.tabWidget.setCurrentWidget(widget)
            self._tabs[name] = widget

            if len(self._tabs) > self.max_tabs:
                msg = f"Maximum number of tabs reached ({self.max_tabs}). Close some tabs to open new ones."
                log.warning(msg)
                return

        elif self.tabWidget.indexOf(self._tabs[name]) == -1:
            msg = f"Tab {name} was closed, reopening."
            log.debug(msg)

            self.tabWidget.addTab(self._tabs[name], name)
        else:
            msg = f"Tab {name} already open, switching to it."
            log.debug(msg)

            self.tabWidget.setCurrentWidget(self._tabs[name])

        self.listView.setModel(self.currentWidget.lmodel)

        if not self.setReferenceButton.isEnabled():
            self.setReferenceButton.setEnabled(True)

    @QtCore.Slot(int)
    def onTabCloseRequested(self, index: int) -> None:
        self.tabWidget.widget(index)
        name = self.tabWidget.tabText(index)

        log.debug(f"Closing tab {name}")

        # keep tab internally, but close the tab
        self.tabWidget.removeTab(index)

        if len(self._tabs) > self.max_tabs:
            self._tabs.pop(name)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # close window only if closed programmatically, otherwise hide
        if event.spontaneous():
            event.accept()
        else:
            self.hide()
            event.ignore()


class ReferenceSelectorDialog(QtWidgets.QDialog, Ui_ReferenceSelectorDialog):
    _model: QtCore.QAbstractProxyModel | None

    def __init__(
        self,
        model: QtCore.QAbstractListModel | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setupUi(self)

        if model is not None:
            proxy_model = QtCore.QIdentityProxyModel()
            proxy_model.setSourceModel(model)
            self._model = proxy_model
        else:
            self._model = None

        self.listView.setModel(self._model)
        self.setWindowTitle("Select Reference Cycle")

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def _get_model(self) -> QtCore.QAbstractProxyModel:
        if self._model is None:
            msg = "Model has not been set."
            raise ValueError(msg)
        return self._model

    def _set_model(self, model: QtCore.QAbstractListModel) -> None:
        proxy_model = QtCore.QIdentityProxyModel()
        proxy_model.setSourceModel(model)
        self._model = proxy_model
        self.listView.setModel(self._model)

    model = property(_get_model, _set_model)

    @property
    def selected_item(self) -> QtCore.QModelIndex | None:
        try:
            return self.listView.selectedIndexes()[0]
        except IndexError:
            return None
