"""Offline figures for discrete RAGTruth confidence thresholds; no fitted curves."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

import experiment_v2 as shared

COLORS = ('#2878b5', '#e58b23', '#359764')


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def validation_plots(out, points, folds, metrics):
    choices = json.loads((out / 'selected_threshold.json').read_text())['models']
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout='constrained')
    for col, ((alias, model), color) in enumerate(zip(shared.LLMS.items(), COLORS)):
        series = [r for r in points if r['model'] == model and r['confidence_threshold'] is not None]
        x = [r['confidence_threshold'] for r in series]
        for metric, marker, label in (('hallucination_f1', 'o', 'H F1'), ('accuracy', '^', 'Accuracy')):
            axes[0, col].scatter(x, [r[metric]*100 for r in series], marker=marker, s=24, label='Cascade '+label)
            axes[0, col].axhline(metrics['jev_only'][metric]*100, color='gray', ls='--', label='JEV '+label,
                                alpha=1 if marker == 'o' else .5)
            axes[0, col].axhline(metrics['llm_only/'+alias][metric]*100, color=color, ls=':', label='LLM '+label,
                                alpha=1 if marker == 'o' else .5)
        selected = choices[model]['confidence_threshold']
        axes[0, col].set_title(alias.capitalize() + (' | selected: no routing' if selected is None else f' | selected threshold: {selected:g}'))
        axes[0, col].set_ylabel('Validation score (%)')
        axes[0, col].legend(fontsize=7, ncol=2)
        axes[1, col].scatter(x, [r['routing_rate']*100 for r in series], color=color, s=24)
        axes[1, col].axhline(20, color='gray', ls='--', label='Selection budget')
        axes[1, col].set_ylabel('Routed (%)')
        axes[1, col].set_ylim(bottom=0)
        for field, marker, label in (('corrected', 'o', 'Corrected'), ('harmed', 'x', 'New errors'), ('net', '^', 'Net')):
            axes[2, col].scatter(x, [r[field] for r in series], marker=marker, s=24, label=label)
        axes[2, col].axhline(0, color='gray', ls='--')
        axes[2, col].set_ylabel('Responses vs JEV')
        axes[2, col].legend(fontsize=8)
        for ax in axes[:, col]:
            ax.set_xlabel('Observed confidence threshold')
            ax.grid(alpha=.2)
    fig.suptitle('RAGTruth QA validation | 1,157 responses / 200 sources\nDiscrete candidates; no routing is the JEV baseline; selection objective is H F1')
    save(fig, out / 'threshold_curve.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    for offset, ((alias, model), color) in enumerate(zip(shared.LLMS.items(), COLORS)):
        rows = [r for r in folds if r['model'] == model]
        x = [r['fold']+(offset-1)*.16 for r in rows]
        for ax, key in zip(axes[:2], ('delta_h_f1', 'delta_accuracy')):
            ax.scatter(x, [r[key]*100 for r in rows], color=color, label=alias, s=45)
            ax.axhline(0, color='gray', ls='--', alpha=.4)
        axes[2].scatter(x, [-.04 if r['confidence_threshold'] is None else r['confidence_threshold'] for r in rows],
                        color=color, label=alias, s=45)
    axes[0].set_ylabel('Held-out H F1 gain vs JEV (pp)')
    axes[1].set_ylabel('Held-out accuracy gain vs JEV (pp)')
    axes[2].set_ylabel('Threshold selected on four other folds')
    axes[2].axhline(-.04, color='gray', ls=':', label='No routing (None)')
    axes[2].annotate('None', (5.25, -.04), fontsize=8)
    for ax in axes:
        ax.set_xlabel('Held-out source-group fold')
        ax.set_xticks(range(1, 6))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Five-fold validation | fixed source groups; thresholds never selected on held-out responses')
    save(fig, out / 'cross_validation.png')


def final_plots(out, rows, metrics, intervals):
    from plot_ragtruth_costs import write_plot
    write_plot(out)
    names = [r['scenario'] for r in rows]
    short = [n.replace('llm_only/', '').replace('cascade/', 'JEV + ').replace('jev_only', 'JEV') for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    x = np.arange(len(rows))
    axes[0, 0].bar(x-.18, [r['accuracy']*100 for r in rows], .36, label='Accuracy')
    axes[0, 0].bar(x+.18, [r['hallucination_f1']*100 for r in rows], .36, label='H F1')
    axes[0, 0].set_xticks(x, short, rotation=30, ha='right')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_ylabel('Score (%)')
    axes[0, 0].legend()
    for ax, metric in ((axes[0, 1], 'hallucination_f1'), (axes[1, 0], 'accuracy')):
        data = [intervals[n][metric] for n in names[1:]]
        for i, r in enumerate(data):
            ax.plot([r['ci95_low']*100, r['ci95_high']*100], [i, i], color='#2878b5')
            ax.scatter(r['difference']*100, i, color='#2878b5')
        ax.set_yticks(range(len(data)), short[1:])
        ax.axvline(0, color='gray', ls='--')
        ax.set_xlabel(f'{metric} difference vs JEV (pp), 95% interval')
    cascades = [r for r in rows if r['scenario'].startswith('cascade/')]
    x = np.arange(3)
    axes[1, 1].bar(x-.18, [r['corrected'] for r in cascades], .36, label='Corrected')
    axes[1, 1].bar(x+.18, [r['harmed'] for r in cascades], .36, label='New errors')
    axes[1, 1].set_xticks(x, [f"{r['scenario'].split('/')[1]}\n{r['routed']}/{r['samples']} routed ({r['routing_rate']:.1%})" for r in cascades])
    axes[1, 1].set_ylabel('Responses')
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(axis='x' if ax in (axes[0, 1], axes[1, 0]) else 'y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('RAGTruth QA independent test | 875 responses / 150 sources\nFrozen validation thresholds; 10,000 paired source bootstrap resamples')
    save(fig, out / 'final_results.png')
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), layout='constrained')
    for ax, name, label in zip(axes.flat, names, short):
        values = np.array([[metrics[name]['confusion'][g][p] for p in ('H', 'N', 'INVALID')] for g in ('H', 'N')])
        ax.imshow(values, cmap='Blues', vmin=0, vmax=max(1, values.max()))
        for (r, c), value in np.ndenumerate(values):
            ax.text(c, r, str(value), ha='center', va='center', color='white' if value > values.max()/2 else 'black')
        ax.set_xticks(range(3), ['H', 'N', 'Invalid'])
        ax.set_yticks(range(2), ['H', 'N'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Gold')
        ax.set_title(label)
    for ax in list(axes.flat)[len(names):]:
        ax.axis('off')
    fig.suptitle('RAGTruth QA test confusion counts | original strict H/N gold')
    save(fig, out / 'confusion_matrices.png')
