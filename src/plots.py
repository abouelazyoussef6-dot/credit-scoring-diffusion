"""Style commun à toutes les figures du projet."""
import matplotlib
import matplotlib.pyplot as plt

# Palette catégorielle à ordre fixe (distinguable en cas de daltonisme)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY = "#8a8a8a"

C_GOOD, C_BAD = BLUE, ORANGE               # une couleur par classe, partout
METHOD_COLORS = {                           # courbes : une couleur par méthode, partout
    "Aucune": GREY, "Pondération": VIOLET, "SMOTE": BLUE,
    "TVAE": YELLOW, "CTGAN": ORANGE, "TabDDPM": AQUA,
}
# Graphiques où chaque ligne porte déjà le nom de la méthode : la couleur indique la famille
FAMILIES = {
    "Sans création de données": (VIOLET, ["Pondération", "Sous-échantillonnage"]),
    "Sur-échantillonnage classique": (BLUE, ["Duplication", "SMOTE"]),
    "Générateurs profonds": (ORANGE, ["TVAE", "CTGAN", "TabDDPM"]),
}
FAMILY_COLOR = {m: color for color, methods in FAMILIES.values() for m in methods}
FAMILY_COLOR["Aucune"] = GREY
METHOD_MARKERS = {"Aucune": "o", "SMOTE": "o", "TVAE": "D", "CTGAN": "s", "TabDDPM": "^",
                  "Pondération": "v", "Duplication": "P", "Sous-échantillonnage": "X"}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
        "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA,
                                                    GREEN, VIOLET, RED]),
    })
