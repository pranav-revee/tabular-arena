export const churnData = {
  dataset: 'telco_churn',
  n_samples: 7043,
  n_features: 19,
  target_rate: 0.265,
  models: [
    {
      name: 'LightGBM (Default)',
      category: 'gradient_boosting',
      tuned: false,
      metrics: {
        auc_roc: 0.8337,
        log_loss: 0.4345,
        train_time_sec: 0.45,
        inference_time_ms_per_1k: 1.1,
        peak_memory_mb: 9.5,
      },
      scaling: [
        { n_samples: 500, auc: 0.7926 },
        { n_samples: 1000, auc: 0.8098 },
        { n_samples: 2000, auc: 0.8142 },
        { n_samples: 3500, auc: 0.823 },
        { n_samples: 5600, auc: 0.8353 },
      ],
    },
    {
      name: 'LightGBM (Tuned)',
      category: 'gradient_boosting',
      tuned: true,
      metrics: {
        auc_roc: 0.8214,
        log_loss: 0.4832,
        train_time_sec: 31.37,
        inference_time_ms_per_1k: 9.3,
        peak_memory_mb: 14.3,
      },
      scaling: [
        { n_samples: 500, auc: 0.8128 },
        { n_samples: 1000, auc: 0.8155 },
        { n_samples: 2000, auc: 0.8095 },
        { n_samples: 3500, auc: 0.815 },
        { n_samples: 5600, auc: 0.8152 },
      ],
    },
    {
      name: 'CatBoost (Default)',
      category: 'gradient_boosting',
      tuned: false,
      metrics: {
        auc_roc: 0.8436,
        log_loss: 0.4201,
        train_time_sec: 1.9,
        inference_time_ms_per_1k: 2.1,
        peak_memory_mb: 152.5,
      },
      scaling: [
        { n_samples: 500, auc: 0.8277 },
        { n_samples: 1000, auc: 0.8324 },
        { n_samples: 2000, auc: 0.8375 },
        { n_samples: 3500, auc: 0.8419 },
        { n_samples: 5600, auc: 0.8447 },
      ],
    },
    {
      name: 'CatBoost (Tuned)',
      category: 'gradient_boosting',
      tuned: true,
      metrics: {
        auc_roc: 0.8313,
        log_loss: 0.4601,
        train_time_sec: 59.07,
        inference_time_ms_per_1k: 1.6,
        peak_memory_mb: 1.3,
      },
      scaling: [
        { n_samples: 500, auc: 0.8028 },
        { n_samples: 1000, auc: 0.8061 },
        { n_samples: 2000, auc: 0.8126 },
        { n_samples: 3500, auc: 0.8232 },
        { n_samples: 5600, auc: 0.8292 },
      ],
    },
    {
      name: 'XGBoost (Default)',
      category: 'gradient_boosting',
      tuned: false,
      metrics: {
        auc_roc: 0.8187,
        log_loss: 0.4691,
        train_time_sec: 0.14,
        inference_time_ms_per_1k: 2.0,
        peak_memory_mb: 7.6,
      },
      scaling: [
        { n_samples: 500, auc: 0.7817 },
        { n_samples: 1000, auc: 0.7993 },
        { n_samples: 2000, auc: 0.8024 },
        { n_samples: 3500, auc: 0.8058 },
        { n_samples: 5600, auc: 0.8207 },
      ],
    },
    {
      name: 'XGBoost (Tuned)',
      category: 'gradient_boosting',
      tuned: true,
      metrics: {
        auc_roc: 0.8383,
        log_loss: 0.4235,
        train_time_sec: 12.43,
        inference_time_ms_per_1k: 1.6,
        peak_memory_mb: 7.6,
      },
      scaling: [
        { n_samples: 500, auc: 0.7941 },
        { n_samples: 1000, auc: 0.8052 },
        { n_samples: 2000, auc: 0.8182 },
        { n_samples: 3500, auc: 0.8326 },
        { n_samples: 5600, auc: 0.8416 },
      ],
    },
    {
      name: 'AutoGluon',
      category: 'automl',
      tuned: false,
      metrics: {
        auc_roc: 0.8445,
        log_loss: 0.4174,
        train_time_sec: 9.54,
        inference_time_ms_per_1k: 13.5,
        peak_memory_mb: 388.5,
      },
      scaling: [
        { n_samples: 500, auc: 0.8289 },
        { n_samples: 1000, auc: 0.8342 },
        { n_samples: 2000, auc: 0.8401 },
        { n_samples: 3500, auc: 0.841 },
        { n_samples: 5600, auc: 0.844 },
      ],
    },
  ],
}

export const MODEL_COLORS: Record<string, string> = {
  'LightGBM (Default)': '#10b981',
  'LightGBM (Tuned)': '#34d399',
  'CatBoost (Default)': '#22c55e',
  'CatBoost (Tuned)': '#4ade80',
  'XGBoost (Default)': '#6366f1',
  'XGBoost (Tuned)': '#818cf8',
  AutoGluon: '#f59e0b',
  TabPFN: '#f43f5e',
}

export const CATEGORY_COLORS: Record<string, string> = {
  gradient_boosting: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  deep_learning: 'text-sky-400 bg-sky-500/10 border-sky-500/20',
  automl: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  foundation_model: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
}

export const CATEGORY_LABELS: Record<string, string> = {
  gradient_boosting: 'Gradient Boosting',
  deep_learning: 'Deep Learning',
  automl: 'AutoML',
  foundation_model: 'Foundation Model',
}
