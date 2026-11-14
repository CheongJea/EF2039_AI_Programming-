import matplotlib.pyplot as plt
import json

# make figure of test loss on web result based on json files
if __name__ == "__main__":
    
    with open("./results/web_result.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 4.5), gridspec_kw={'height_ratios': [1, 3.5], 'hspace': 0.08})

    for ax in (ax_top, ax_bottom):
        
        marker_l = ["o", "^", "x"]
        for i in range(len(["Q1.1", "Q1.2 & Q3.2", "Q1.3"])):
            case = ["Q1.1", "Q1.2 & Q3.2", "Q1.3"][i]
            epoch_l = data[case]["epochs"]
            test_loss_l = data[case]["test_loss"]
            ax.plot(epoch_l, test_loss_l, label=case, color="black", marker=marker_l[i], markersize=10)
        
        
        marker_l = ["s", "*"]
        for i in range(len(["Q2.1", "Q2.2"])):
            case = ["Q2.1", "Q2.2"][i]
            epoch_l = data[case]["epochs"]
            test_loss_l = data[case]["test_loss"]
            ax.plot(epoch_l, test_loss_l, label=case, color="red", marker=marker_l[i], markersize=10, linestyle="--", markeredgecolor="black", markeredgewidth=1.2)

        
        marker_l = ["v"]
        for i in range(len(["Q3.3"])):
            case = ["Q3.3"][i]
            epoch_l = data[case]["epochs"]
            test_loss_l = data[case]["test_loss"]
            ax.plot(epoch_l, test_loss_l, label=case, color="blue", marker=marker_l[i], markersize=10, linestyle=":", markeredgecolor="black", markeredgewidth=1.2)
        
        color_l = ["#8100FF"]
        marker_l = ["D"]
        for i in range(len(["Q3.1"])):
            case = ["Q3.1"][i]
            epoch_l = data[case]["epochs"]
            test_loss_l = data[case]["test_loss"]
            ax.plot(epoch_l, test_loss_l, label=case, color=color_l[i], marker=marker_l[i], markersize=10, linestyle=":", markeredgecolor="black", markeredgewidth=1.2)

    ax_top.set_ylim(4.5, 4.7)
    ax_bottom.set_ylim(0.15, 0.95)

    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False)

    kwargs = dict(marker=r'$\sim$', markersize=14, linestyle="none", color='k', mec='k', mew=1.2, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    ax_top.legend(frameon=False, fontsize=15, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax_bottom.set_xlabel("Epochs")
    fig.supylabel("Test loss", x=0.03)

    for ax in (ax_top, ax_bottom):
        for spines in ax.spines.values():
            spines.set_linewidth(2)

    plt.savefig("./results/01_testloss.png", dpi=500, bbox_inches="tight")