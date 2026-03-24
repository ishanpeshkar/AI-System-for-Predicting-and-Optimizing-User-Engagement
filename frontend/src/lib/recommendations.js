export const API_URL = 'http://127.0.0.1:8000/api/recommendations'

export const MODEL_DEFINITIONS = [
  {
    key: 'graph',
    badge: 'M1',
    name: 'Graph (Node2Vec)',
    algorithm: 'Graph (Node2Vec)',
    shortDescription: 'Relationship embeddings across movie graph nodes',
    scoreLabel: 'cosine similarity',
    accent: '#79c0ff',
    metric: 'Metric: 64-dim embedding space',
  },
  {
    key: 'baseline',
    badge: 'M2',
    name: 'Baseline (RF)',
    algorithm: 'Baseline (RF)',
    shortDescription: 'Tabular RF model estimating engagement probability',
    scoreLabel: 'engagement prob',
    accent: '#3fb950',
    metric: 'Metric: AUC 0.74 (single-model baseline)',
  },
  {
    key: 'lstm',
    badge: 'M3',
    name: 'Sequential (LSTM)',
    algorithm: 'Sequential (LSTM)',
    shortDescription: 'Sequence-aware model forecasting next-watch intent',
    scoreLabel: 'next-watch prob',
    accent: '#58a6ff',
    metric: 'Metric: HR@10 8.14%',
  },
  {
    key: 'rl',
    badge: 'M4',
    name: 'RL (Bandit)',
    algorithm: 'RL (Bandit)',
    shortDescription: 'Contextual bandit balancing explore and exploit',
    scoreLabel: 'bandit reward',
    accent: '#d2a534',
    metric: 'Metric: regret reduced by 61%',
  },
]

export const WATCH_HISTORY = {
  1: [
    'Toy Story',
    'GoodFellas',
    'Pulp Fiction',
    'Batman',
    'True Lies',
    'Apollo 13',
    'Braveheart',
    'Seven',
    'Usual Suspects',
    'Heat',
  ],
  5: [
    'Casablanca',
    'Rear Window',
    'Psycho',
    'Vertigo',
    'North by Northwest',
    'Citizen Kane',
    'Sunset Blvd',
    'Double Indemnity',
    'Maltese Falcon',
    'Laura',
  ],
}

export const USER_STATS = {
  1: '610 ratings · avg 3.9',
  5: '342 ratings · avg 3.6',
}

export function getUserHistory(userId) {
  return WATCH_HISTORY[userId] ?? WATCH_HISTORY[1]
}

export function getUserStats(userId) {
  return USER_STATS[userId] ?? 'Ratings profile unavailable'
}

export async function postRecommendations({ userId, topK, algorithm }) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, topK, algorithm }),
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Request failed (${response.status}): ${errorText}`)
  }

  const data = await response.json()
  return data.recommendations ?? []
}

export function formatScore(score) {
  return typeof score === 'number' ? score.toFixed(3) : '--'
}

export function splitGenres(genres) {
  if (!genres || typeof genres !== 'string') {
    return []
  }
  return genres.split('|').slice(0, 3)
}
