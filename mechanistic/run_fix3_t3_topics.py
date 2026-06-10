"""
Fix 3: Topic-category labels for T3 questions.
Assign semantic topic categories to all 29 T3 questions using keyword matching on question text.
Update the failure-mode scatter (Gap B) with topic-colored points.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings('ignore')

BASE    = '/shared/storage-01/users/zahraei2/mena_normal'
RESULTS = os.path.join(BASE, 'results')

# ── Load question text and tier classification ────────────────────────────────
tiers_df = pd.read_csv(os.path.join(BASE, 'tiers.csv'))
t3_df    = tiers_df[tiers_df['Tier'] == 3][['question_id', 'No Mention']].copy()
t3_df.columns = ['question_id', 'text']
print(f'T3 questions: {len(t3_df)}')

# ── Keyword-based topic assignment ────────────────────────────────────────────
def assign_topic(text):
    t = text.lower()
    if any(w in t for w in ['homosexual', 'gay', 'same-sex', 'lgb']):
        return 'LGBTQ+'
    if any(w in t for w in ['beat', 'violence', 'abuse']):
        return 'Domestic Violence'
    if any(w in t for w in ['women', 'woman', 'female', 'girl', 'wife', 'income than']):
        return 'Gender Equality'
    if any(w in t for w in ['religion', 'religious', 'faith', 'church', 'mosque', 'god']):
        return 'Religious Tolerance'
    if any(w in t for w in ['race', 'racial', 'ethnic', 'immigrant', 'foreign', 'different race']):
        return 'Racial/Ethnic Tolerance'
    if any(w in t for w in ['drug', 'aids', 'hiv', 'disease']):
        return 'Social Stigma'
    if any(w in t for w in ['political', 'leader', 'government', 'democrat', 'authorit']):
        return 'Political'
    if any(w in t for w in ['men make better', 'men should have', 'boy than', 'business executive']):
        return 'Gender Equality'
    return 'Other'

t3_df['topic'] = t3_df['text'].apply(assign_topic)
print('\nTopic distribution:')
print(t3_df.groupby('topic')['question_id'].count().to_string())
print('\nAll T3 questions with topics:')
for _, row in t3_df.iterrows():
    print(f"  Q{row['question_id']} [{row['topic']}]: {row['text'][:90]}")

# ── Load failure mode data from Gap B ─────────────────────────────────────────
q_stats = pd.read_csv(os.path.join(RESULTS, 'gapB_t3_question_taxonomy.csv'))
q_stats = q_stats.merge(t3_df[['question_id', 'topic', 'text']], on='question_id', how='left')
q_stats['topic'] = q_stats['topic'].fillna('Other')
q_stats.to_csv(os.path.join(RESULTS, 'fix3_t3_topics.csv'), index=False)
print(f'\nSaved: fix3_t3_topics.csv')

# ── Updated failure-mode figure with topic colors ─────────────────────────────
TOPIC_COLORS = {
    'LGBTQ+':              '#E53935',
    'Gender Equality':     '#8E24AA',
    'Domestic Violence':   '#D84315',
    'Religious Tolerance': '#1565C0',
    'Racial/Ethnic Tolerance': '#00838F',
    'Social Stigma':       '#558B2F',
    'Political':           '#EF6C00',
    'Other':               '#78909C',
}
TOPIC_MARKERS = {
    'LGBTQ+':              'o',
    'Gender Equality':     's',
    'Domestic Violence':   'D',
    'Religious Tolerance': '^',
    'Racial/Ethnic Tolerance': 'v',
    'Social Stigma':       'P',
    'Political':           'X',
    'Other':               'o',
}

REFUSAL_THRESH = 0.30
NVAS_THRESH    = 0.60

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('white')

for ax_i, (ax, title_suffix) in enumerate(zip(axes, ['by Failure Mode', 'by Topic Category'])):
    ax.set_facecolor('#FAFAFA')

    if ax_i == 0:
        # Original failure mode coloring
        mode_colors = {
            'Suppressed':          '#E67C13',
            'Double failure':      '#B71C1C',
            'Representational':    '#7B1FA2',
            'Well-aligned':        '#2E7D32',
        }
        def failure_mode_short(row):
            hi_ref  = row['refusal_rate'] >= REFUSAL_THRESH
            lo_nvas = row['nvas_when_answered'] < NVAS_THRESH
            if hi_ref and not lo_nvas:  return 'Suppressed'
            if hi_ref and lo_nvas:      return 'Double failure'
            if not hi_ref and lo_nvas:  return 'Representational'
            return 'Well-aligned'
        q_stats['fm'] = q_stats.apply(failure_mode_short, axis=1)
        for fm, grp in q_stats.groupby('fm'):
            ax.scatter(grp['refusal_rate'], grp['nvas_when_answered'],
                       color=mode_colors.get(fm, '#888'), s=90, zorder=5,
                       edgecolors='white', linewidths=0.5,
                       label=f'{fm} (n={len(grp)})')
            for _, row in grp.iterrows():
                ax.annotate(f'Q{row["question_id"]}',
                            (row['refusal_rate'], row['nvas_when_answered']),
                            fontsize=6.5, xytext=(3, 2), textcoords='offset points', color='#333')
    else:
        # Topic coloring
        for topic, grp in q_stats.groupby('topic'):
            ax.scatter(grp['refusal_rate'], grp['nvas_when_answered'],
                       color=TOPIC_COLORS.get(topic, '#888'),
                       marker=TOPIC_MARKERS.get(topic, 'o'),
                       s=100, zorder=5, edgecolors='white', linewidths=0.5,
                       label=f'{topic} (n={len(grp)})')
            for _, row in grp.iterrows():
                ax.annotate(f'Q{row["question_id"]}',
                            (row['refusal_rate'], row['nvas_when_answered']),
                            fontsize=6.5, xytext=(3, 2), textcoords='offset points',
                            color=TOPIC_COLORS.get(row['topic'], '#555'))

    ax.axvline(REFUSAL_THRESH, color='#aaa', lw=1, ls='--', alpha=0.7)
    ax.axhline(NVAS_THRESH,    color='#aaa', lw=1, ls='--', alpha=0.7)

    # Quadrant labels
    for (tx, ty, label) in [(0.72, 0.88, 'Suppressed\n(knows, hides)'),
                             (0.72, 0.12, 'Double failure'),
                             (0.10, 0.12, 'Representational\nbias'),
                             (0.10, 0.88, 'Well-aligned')]:
        ax.text(tx, ty, label, transform=ax.transAxes, fontsize=7.5,
                ha='center', va='center', color='#888', alpha=0.6)

    ax.set_xlabel('Refusal Rate (Persona EN, instruct models)', fontsize=9.5)
    ax.set_ylabel('NVAS when Answered', fontsize=9.5)
    ax.set_title(f'T3 Question Taxonomy {title_suffix}', fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9, loc='upper left',
              ncol=1 if ax_i == 0 else 2)
    ax.grid(lw=0.3, alpha=0.4, color='#ccc')
    ax.set_xlim(-0.02, 0.65); ax.set_ylim(0.10, 1.0)

# Add insight annotation
fig.text(0.5, -0.02,
         'LGBTQ+ questions cluster in top-right (suppressed — refused but accurate when answered).\n'
         'Gender equality questions cluster in lower-left (representational bias — answered but wrong).',
         ha='center', fontsize=9, color='#444', style='italic')

fig.suptitle('T3 Question Taxonomy: Failure Mode and Topic Category\n'
             'LGBTQ+ → suppression (prompt fix works)  |  Gender equality → representational bias (training fix needed)',
             fontsize=10.5, y=1.02)
fig.tight_layout()
out = os.path.join(RESULTS, 'fix3_t3_topics.pdf')
fig.savefig(out, bbox_inches='tight', dpi=200); plt.close(fig)
print(f'Saved: {out}')

# ── LaTeX topic table for report ───────────────────────────────────────────────
topic_summary = q_stats.groupby('topic').agg(
    n=('question_id', 'count'),
    mean_refusal=('refusal_rate', 'mean'),
    mean_nvas=('nvas_when_answered', 'mean'),
).reset_index().sort_values('mean_refusal', ascending=False)
print('\nTopic summary:')
print(topic_summary.round(3).to_string())

latex_rows = []
for _, row in topic_summary.iterrows():
    latex_rows.append(
        f'{row["topic"]} & {int(row["n"])} & {row["mean_refusal"]:.2f} & {row["mean_nvas"]:.2f} \\\\')

with open(os.path.join(RESULTS, 'fix3_t3_topic_table.tex'), 'w') as f:
    f.write(r"""\begin{table}[H]
\centering\small
\caption{T3 questions by semantic topic category. Mean refusal rate and NVAS-when-answered
averaged across instruct models (Persona EN). Topics with high refusal and high NVAS are
suppression failures (addressable by Third-person framing); topics with low NVAS regardless of
refusal rate are representational bias failures (require training-time intervention).}
\label{tab:t3_topics}
\begin{tabular}{lrcc}
\toprule
Topic & $n$ & Mean Refusal & Mean NVAS (answered) \\
\midrule
""")
    f.write('\n'.join(latex_rows))
    f.write(r"""
\bottomrule
\end{tabular}
\end{table}""")
print('Saved: fix3_t3_topic_table.tex')
print('\nDone.')
