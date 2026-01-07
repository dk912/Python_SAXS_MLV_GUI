import sys
import json
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QLabel, QComboBox, QMessageBox, QCheckBox, QTabWidget
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from scipy.optimize import least_squares

from model.intensity import intensity
from model.electron_density import electron_density


# ============================================================
# Matplotlib canvas
# ============================================================
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, xlabel="", ylabel="", loglog=True):
        fig = Figure(figsize=(6, 4))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)

        if loglog:
            self.ax.set_xscale("log")
            self.ax.set_yscale("log")

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)


# ============================================================
# Main GUI
# ============================================================
class LamellarGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lamellar SAXS / Reflectivity Model")

        self.Qz = None
        self.Iexp = None
        self.Ierr = None

        self.init_ui()

    # --------------------------------------------------------
    def init_ui(self):

        main = QHBoxLayout(self)

        # ================= LEFT PANEL =================
        left = QVBoxLayout()

        # ---- Load data ----
        load_btn = QPushButton("Load q–I–σ data")
        load_btn.clicked.connect(self.load_data)
        left.addWidget(load_btn)

        # ---- Q units ----
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("Q units:"))
        self.q_unit = QComboBox()
        self.q_unit.addItems(["Å⁻¹", "nm⁻¹"])
        unit_layout.addWidget(self.q_unit)
        left.addLayout(unit_layout)

        # ---- Parameter table ----
        self.param_table = QTableWidget(10, 6)
        self.param_table.setHorizontalHeaderLabels(
            ["Fix", "Parameter", "Initial", "Fitted", "Lower", "Upper"]
        )

        names = ["rhoH", "zh", "sigmaH", "sigmaC",
                 "d", "N", "eta", "Ndiff", "rhoC", "A"]

        defaults = [0.45, 20.98, 2.53, 8.2,
                    78.83, 22, 0.13, 4.5, 0.24, 0.058]

        lower = [0.1, 5, 0.5, 1,
                 40, 5, 0.01, 0, 0.05, 0]

        upper = [1.0, 40, 10, 20,
                 120, 50, 1.0, 20, 0.5, 0.5]

        for i in range(10):
            chk = QCheckBox()
            self.param_table.setCellWidget(i, 0, chk)
            self.param_table.setItem(i, 1, QTableWidgetItem(names[i]))
            self.param_table.setItem(i, 2, QTableWidgetItem(str(defaults[i])))
            self.param_table.setItem(i, 3, QTableWidgetItem(str(defaults[i])))
            self.param_table.setItem(i, 4, QTableWidgetItem(str(lower[i])))
            self.param_table.setItem(i, 5, QTableWidgetItem(str(upper[i])))

        left.addWidget(self.param_table)

        # ---- Buttons ----
        eval_btn = QPushButton("Evaluate (Initial)")
        eval_btn.clicked.connect(self.evaluate_model)
        left.addWidget(eval_btn)

        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.run_fit)
        left.addWidget(fit_btn)

        save_btn = QPushButton("Save parameters")
        save_btn.clicked.connect(self.save_parameters)
        left.addWidget(save_btn)

        loadp_btn = QPushButton("Load parameters")
        loadp_btn.clicked.connect(self.load_parameters)
        left.addWidget(loadp_btn)

        self.status_label = QLabel("Status: Idle")
        left.addWidget(self.status_label)

        self.r2_label = QLabel("R² : ---")
        left.addWidget(self.r2_label)

        main.addLayout(left)

        # ================= RIGHT PANEL =================
        self.tabs = QTabWidget()

        self.canvas_I = MplCanvas("Q (Å⁻¹)", "I(q)", loglog=True)
        self.canvas_res = MplCanvas("Q (Å⁻¹)", "Residuals", loglog=False)
        self.canvas_ed = MplCanvas("z (Å)", "Electron density", loglog=False)

        self.tabs.addTab(self.canvas_I, "Intensity")
        self.tabs.addTab(self.canvas_res, "Residuals")
        self.tabs.addTab(self.canvas_ed, "ED(z)")

        main.addWidget(self.tabs)

    # --------------------------------------------------------
    def load_data(self):

        file, _ = QFileDialog.getOpenFileName(
            self, "Load Data", "", "*.dat *.txt *.csv"
        )
        if not file:
            return

        data = pd.read_csv(file, delim_whitespace=True, header=None)

        Q = data.iloc[:, 0].values
        I = data.iloc[:, 1].values

        if self.q_unit.currentText() == "nm⁻¹":
            Q = Q * 0.1  # nm⁻¹ → Å⁻¹

        self.Qz = Q
        self.Iexp = I / np.max(I)

        if data.shape[1] > 2:
            self.Ierr = data.iloc[:, 2].values
        else:
            self.Ierr = np.ones_like(self.Iexp)

        self.canvas_I.ax.clear()
        self.canvas_I.ax.loglog(self.Qz, self.Iexp, 'ko', label="Data")
        self.canvas_I.legend = self.canvas_I.ax.legend()
        self.canvas_I.draw()

    # --------------------------------------------------------
    def get_parameters_and_bounds(self):

        p0, lb, ub, free_idx = [], [], [], []

        for i in range(self.param_table.rowCount()):
            p0.append(float(self.param_table.item(i, 2).text()))
            lb.append(float(self.param_table.item(i, 4).text()))
            ub.append(float(self.param_table.item(i, 5).text()))

            fixed = self.param_table.cellWidget(i, 0).isChecked()
            if not fixed:
                free_idx.append(i)

        return np.array(p0), np.array(lb), np.array(ub), free_idx

    # --------------------------------------------------------
    def evaluate_model(self):

        if self.Qz is None:
            return

        p0 = np.array([
            float(self.param_table.item(i, 2).text())
            for i in range(self.param_table.rowCount())
        ])

        Imodel = intensity(p0, self.Qz)
        self.update_plots(Imodel, p0)

    # --------------------------------------------------------
    def run_fit(self):

        if self.Qz is None:
            return

        self.status_label.setText("Status: Fitting…")
        QApplication.processEvents()

        p0, lb, ub, free_idx = self.get_parameters_and_bounds()

        # enforce feasibility
        for i in free_idx:
            p0[i] = np.clip(p0[i], lb[i], ub[i])

        def assemble(pfree):
            p = p0.copy()
            p[free_idx] = pfree
            return p

        def residual(pfree):
            return self.Iexp - intensity(assemble(pfree), self.Qz)

        res = least_squares(
            residual,
            p0[free_idx],
            bounds=(lb[free_idx], ub[free_idx])
        )

        pfit = assemble(res.x)

        for i, val in enumerate(pfit):
            self.param_table.setItem(
                i, 3, QTableWidgetItem(f"{val:.6g}")
            )

        Imodel = intensity(pfit, self.Qz)
        self.update_plots(Imodel, pfit)

        self.status_label.setText("Status: Done")

    # --------------------------------------------------------
    def update_plots(self, Imodel, parm):

        ss_res = np.sum((self.Iexp - Imodel)**2)
        ss_tot = np.sum((self.Iexp - np.mean(self.Iexp))**2)
        r2 = 1 - ss_res / ss_tot
        self.r2_label.setText(f"R² : {r2:.6f}")

        # Intensity
        self.canvas_I.ax.clear()
        self.canvas_I.ax.loglog(self.Qz, self.Iexp, 'ko', label="Data")
        self.canvas_I.ax.loglog(self.Qz, Imodel, '-r', lw=2, label="Model")
        self.canvas_I.ax.legend()
        self.canvas_I.draw()

        # Residuals
        self.canvas_res.ax.clear()
        self.canvas_res.ax.plot(self.Qz, self.Iexp - Imodel, 'k.')
        self.canvas_res.ax.axhline(0, color='r', lw=1)
        self.canvas_res.draw()

        # ED(z)
        z, rho = electron_density(parm)
        self.canvas_ed.ax.clear()
        self.canvas_ed.ax.plot(z, rho, '-b')
        self.canvas_ed.draw()

    # --------------------------------------------------------
    def save_parameters(self):

        file, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "*.json"
        )
        if not file:
            return

        params = {}
        for i in range(self.param_table.rowCount()):
            name = self.param_table.item(i, 1).text()
            params[name] = {
                "initial": float(self.param_table.item(i, 2).text()),
                "fitted": float(self.param_table.item(i, 3).text()),
                "lower": float(self.param_table.item(i, 4).text()),
                "upper": float(self.param_table.item(i, 5).text()),
                "fixed": self.param_table.cellWidget(i, 0).isChecked()
            }

        with open(file, "w") as f:
            json.dump(params, f, indent=2)

    # --------------------------------------------------------
    def load_parameters(self):

        file, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "*.json"
        )
        if not file:
            return

        with open(file) as f:
            params = json.load(f)

        for i, name in enumerate(params):
            self.param_table.item(i, 2).setText(str(params[name]["initial"]))
            self.param_table.item(i, 3).setText(str(params[name]["fitted"]))
            self.param_table.item(i, 4).setText(str(params[name]["lower"]))
            self.param_table.item(i, 5).setText(str(params[name]["upper"]))
            self.param_table.cellWidget(i, 0).setChecked(params[name]["fixed"])


# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = LamellarGUI()
    gui.show()
    sys.exit(app.exec())
