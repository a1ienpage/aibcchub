#!/usr/bin/env python3
"""
HELIOS MASTER PIPELINE v2.0 - Система персональных рекомендаций для победы в хакатоне
Архитектура: CFS (Feature Store) → QRE (Recommendation Engine) → CGM (Generative Module)
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
import argparse
import sys

warnings.filterwarnings('ignore')

class ClientFeatureStore:
    """Продвинутое хранилище клиентских признаков с нормализацией"""
    
    def __init__(self):
        self.feature_importance = {}
    
    def create_advanced_features(self, data_folder):
        print("🔧 ЭТАП 1: Создание хранилища клиентских признаков (CFS)")
        print("=" * 60)
        
        clients_df = pd.read_csv(os.path.join(data_folder, 'clients.csv'), encoding='utf-8')
        print(f"📊 Загружено {len(clients_df)} клиентов")
        
        all_transactions = []
        all_transfers = []
        
        for filename in os.listdir(data_folder):
            filepath = os.path.join(data_folder, filename)
            
            if "transactions_3m" in filename and filename.endswith(".csv"):
                match = re.search(r'client_(\d+)_transactions_3m\.csv', filename)
                if match:
                    client_code = int(match.group(1))
                    df = pd.read_csv(filepath, encoding='utf-8')
                    df['client_code'] = client_code
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['month'] = df['date'].dt.month
                    all_transactions.append(df)
                    
            elif "transfers_3m" in filename and filename.endswith(".csv"):
                match = re.search(r'client_(\d+)_transfers_3m\.csv', filename)
                if match:
                    client_code = int(match.group(1))
                    df = pd.read_csv(filepath, encoding='utf-8')
                    df['client_code'] = client_code
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['month'] = df['date'].dt.month
                    all_transfers.append(df)
        
        if not all_transactions or not all_transfers:
            raise FileNotFoundError("Не найдены файлы транзакций или переводов")
        
        transactions_df = pd.concat(all_transactions, ignore_index=True)
        transfers_df = pd.concat(all_transfers, ignore_index=True)
        
        print(f"💳 Загружено {len(transactions_df):,} транзакций и {len(transfers_df):,} переводов")
        
        features = clients_df.copy()
        
        # One-hot encoding статусов
        status_mapping = {
            'Студент': {'is_student': 1, 'credit_multiplier': 0.5, 'premium_penalty': -2},
            'Зарплатный клиент': {'is_salary_client': 1, 'credit_multiplier': 1.2, 'premium_penalty': 0},
            'Премиальный клиент': {'is_premium_client': 1, 'credit_multiplier': 1.5, 'premium_penalty': 2},
            'Стандартный клиент': {'is_standard_client': 1, 'credit_multiplier': 1.0, 'premium_penalty': 0}
        }
        
        for status, attrs in status_mapping.items():
            mask = features['status'] == status
            for attr, value in attrs.items():
                features.loc[mask, attr] = value
        
        features = features.fillna(0)
        
        # Финансовое здоровье
        income_types = ['salary_in', 'stipend_in', 'family_in', 'cashback_in', 'refund_in', 'invest_in']
        
        income_stats = []
        for client_code in features['client_code']:
            client_transfers = transfers_df[
                (transfers_df['client_code'] == client_code) & 
                (transfers_df['type'].isin(income_types))
            ]
            
            if len(client_transfers) > 0:
                monthly_income = client_transfers.groupby('month')['amount'].sum()
                stats = {
                    'client_code': client_code,
                    'total_income_3m': client_transfers['amount'].sum(),
                    'avg_monthly_income': monthly_income.mean(),
                    'income_stability_cv': monthly_income.std() / (monthly_income.mean() + 1),
                    'income_trend': np.polyfit(range(len(monthly_income)), monthly_income.values, 1)[0] if len(monthly_income) > 1 else 0,
                    'dominant_income_type': client_transfers['type'].mode().iloc[0] if len(client_transfers) > 0 else 'none'
                }
            else:
                stats = {
                    'client_code': client_code,
                    'total_income_3m': 0, 'avg_monthly_income': 0, 
                    'income_stability_cv': 0, 'income_trend': 0, 'dominant_income_type': 'none'
                }
            
            income_stats.append(stats)
        
        income_df = pd.DataFrame(income_stats)
        features = features.merge(income_df, on='client_code', how='left')
        
        # Анализ трат по категориям
        categories_mapping = {
            'Такси': 'taxi', 'Путешествия': 'travel', 'Отели': 'hotels',
            'Кафе и рестораны': 'restaurants', 'Ювелирные украшения': 'jewelry',
            'Косметика и Парфюмерия': 'cosmetics', 'Одежда и обувь': 'clothing',
            'Продукты питания': 'groceries', 'Медицина': 'medicine',
            'Авто': 'auto', 'АЗС': 'gas', 'Спорт': 'sport',
            'Развлечения': 'entertainment', 'Кино': 'cinema',
            'Питомцы': 'pets', 'Книги': 'books', 'Цветы': 'flowers',
            'Едим дома': 'food_delivery', 'Смотрим дома': 'streaming',
            'Играем дома': 'gaming', 'Подарки': 'gifts',
            'Ремонт дома': 'home_repair', 'Мебель': 'furniture', 'Спа и массаж': 'spa'
        }
        
        for rus_cat, eng_cat in categories_mapping.items():
            cat_transactions = transactions_df[transactions_df['category'] == rus_cat]
            if len(cat_transactions) > 0:
                cat_stats = cat_transactions.groupby('client_code').agg({
                    'amount': ['sum', 'count', 'mean', 'std'],
                    'month': lambda x: x.nunique()
                })
                cat_stats.columns = [
                    f'spend_{eng_cat}_3m', f'tx_count_{eng_cat}_3m', 
                    f'avg_tx_{eng_cat}', f'spend_volatility_{eng_cat}', f'months_active_{eng_cat}'
                ]
                features = features.merge(cat_stats, on='client_code', how='left')
        
        # Общие траты
        total_stats = transactions_df.groupby('client_code').agg({
            'amount': ['sum', 'count', 'std'],
            'category': lambda x: x.nunique(),
            'month': lambda x: x.nunique()
        })
        total_stats.columns = ['total_spend_3m', 'total_tx_count', 'spend_volatility', 'categories_used', 'active_months']
        features = features.merge(total_stats, on='client_code', how='left')
        
        # Процентные доли
        for eng_cat in categories_mapping.values():
            spend_col = f'spend_{eng_cat}_3m'
            if spend_col in features.columns:
                features[f'pct_spend_{eng_cat}'] = features[spend_col] / (features['total_spend_3m'] + 1) * 100
        
        # Специальные агрегаты для продуктов
        travel_related = ['taxi', 'travel', 'hotels', 'gas']
        for cat in travel_related:
            if f'spend_{cat}_3m' not in features.columns:
                features[f'spend_{cat}_3m'] = 0
        features['travel_related_spend'] = features[[f'spend_{cat}_3m' for cat in travel_related]].sum(axis=1)
        features['travel_tx_frequency'] = sum([features.get(f'tx_count_{cat}_3m', 0) for cat in travel_related])
        features['travel_consistency'] = sum([features.get(f'months_active_{cat}', 0) for cat in travel_related])
        
        luxury_cats = ['jewelry', 'cosmetics', 'spa', 'restaurants']
        features['luxury_spend'] = sum([features.get(f'spend_{cat}_3m', 0) for cat in luxury_cats])
        features['luxury_frequency'] = sum([features.get(f'tx_count_{cat}_3m', 0) for cat in luxury_cats])
        features['has_luxury_purchases'] = (features['luxury_spend'] > 0).astype(int)
        
        online_cats = ['gaming', 'streaming', 'food_delivery', 'cinema']
        features['online_services_spend'] = sum([features.get(f'spend_{cat}_3m', 0) for cat in online_cats])
        features['online_frequency'] = sum([features.get(f'tx_count_{cat}_3m', 0) for cat in online_cats])
        
        # Валютные операции
        fx_operations = transfers_df[transfers_df['type'].isin(['fx_buy', 'fx_sell'])]
        if len(fx_operations) > 0:
            fx_stats = fx_operations.groupby('client_code').agg({
                'amount': ['count', 'sum', 'mean'],
                'month': lambda x: x.nunique()
            })
            fx_stats.columns = ['fx_activity_count', 'fx_volume_total_3m', 'fx_avg_operation', 'fx_active_months']
            features = features.merge(fx_stats, on='client_code', how='left')
        
        # Анализ валют
        fx_tx = transactions_df[transactions_df['currency'] != 'KZT']
        if len(fx_tx) > 0:
            fx_tx_stats = fx_tx.groupby('client_code').agg({
                'currency': lambda x: x.mode().iloc[0] if len(x) > 0 else 'KZT',
                'amount': ['count', 'sum']
            })
            fx_tx_stats.columns = ['main_fx_currency', 'fx_tx_count', 'fx_tx_volume']
            features = features.merge(fx_tx_stats, on='client_code', how='left')
        
        # Инвестиции
        invest_types = ['invest_out', 'invest_in', 'deposit_topup_out', 'deposit_fx_topup_out', 'gold_buy_out']
        invest_transfers = transfers_df[transfers_df['type'].isin(invest_types)]
        if len(invest_transfers) > 0:
            invest_stats = invest_transfers.groupby('client_code').agg({
                'amount': ['count', 'sum', 'mean'],
                'type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'none'
            })
            invest_stats.columns = ['investment_activity_count', 'investment_volume', 'avg_investment', 'main_investment_type']
            features = features.merge(invest_stats, on='client_code', how='left')
        
        # Долговая нагрузка
        debt_types = ['loan_payment_out', 'cc_repayment_out', 'installment_payment_out']
        debt_transfers = transfers_df[transfers_df['type'].isin(debt_types)]
        if len(debt_transfers) > 0:
            debt_stats = debt_transfers.groupby('client_code').agg({
                'amount': ['sum', 'count', 'mean'],
                'month': lambda x: x.nunique()
            })
            debt_stats.columns = ['debt_payments_3m', 'debt_tx_count', 'avg_debt_payment', 'debt_months']
            features = features.merge(debt_stats, on='client_code', how='left')
        
        # Операции с наличными
        atm_transfers = transfers_df[transfers_df['type'] == 'atm_withdrawal']
        if len(atm_transfers) > 0:
            atm_stats = atm_transfers.groupby('client_code').agg({
                'amount': ['sum', 'count', 'mean'],
                'month': lambda x: x.nunique()
            })
            atm_stats.columns = ['atm_withdrawal_total_3m', 'atm_withdrawal_count', 'avg_atm_withdrawal', 'atm_active_months']
            features = features.merge(atm_stats, on='client_code', how='left')
        
        # P2P переводы
        p2p_out = transfers_df[transfers_df['type'] == 'p2p_out']
        if len(p2p_out) > 0:
            p2p_stats = p2p_out.groupby('client_code').agg({
                'amount': ['sum', 'count', 'mean']
            })
            p2p_stats.columns = ['p2p_out_volume', 'p2p_out_count', 'avg_p2p_out']
            features = features.merge(p2p_stats, on='client_code', how='left')
        
        # Расчетные метрики
        features['disposable_income'] = features['avg_monthly_income'] - (features['total_spend_3m'] / 3)
        features['savings_potential'] = np.maximum(features['disposable_income'], 0)
        features['balance_to_income_ratio'] = features['avg_monthly_balance_KZT'] / (features['avg_monthly_income'] + 1)
        features['spend_to_income_ratio'] = (features['total_spend_3m'] / 3) / (features['avg_monthly_income'] + 1)
        features['debt_to_income_ratio'] = features.get('debt_payments_3m', 0) / (features['total_income_3m'] + 1)
        
        # Сигналы потребности
        features['liquidity_need_signal'] = (
            features.get('atm_withdrawal_count', 0) / 3 + 
            features.get('p2p_out_count', 0) / 3 +
            features['debt_to_income_ratio'] * 10
        )
        
        features['premium_readiness_score'] = (
            np.log1p(features['avg_monthly_balance_KZT']) * 0.3 +
            features['luxury_spend'] / (features['total_spend_3m'] + 1) * 100 * 0.4 +
            features.get('premium_penalty', 0) * 0.3
        )
        
        features['investment_readiness_score'] = (
            features['savings_potential'] * 0.4 +
            features.get('investment_activity_count', 0) * 1000 * 0.4 +
            (30 - features.get('age', 30)) * 50 * 0.2
        )
        
        features = features.fillna(0)
        
        # Нормализация percentile
        numeric_cols = [col for col in features.columns if features[col].dtype in ['float64', 'int64'] and col not in ['client_code', 'age']]
        
        for col in numeric_cols:
            if features[col].max() > 0:
                features[f'{col}_percentile'] = features[col].rank(pct=True) * 100
        
        print(f"✅ Создано {len(features.columns)} признаков для {len(features)} клиентов")
        return features


class QuantitativeRecommendationEngine:
    """Движок рекомендаций с адаптивными весами"""
    
    def __init__(self):
        self.scoring_matrix = {
            'Карта для путешествий': {
                'travel_related_spend': 15,
                'travel_tx_frequency': 12,
                'travel_consistency': 10,
                'pct_spend_travel': 13,
                'pct_spend_taxi': 11,
                'pct_spend_hotels': 9,
                'pct_spend_gas': 7,
                'fx_tx_count': 6,
                'age': 2,
                'is_student': -5,
                'avg_monthly_balance_KZT_percentile': 4,
                'disposable_income': 3
            },
            'Премиальная карта': {
                'avg_monthly_balance_KZT_percentile': 20,
                'luxury_spend': 15,
                'luxury_frequency': 10,
                'premium_readiness_score': 18,
                'pct_spend_restaurants': 12,
                'pct_spend_jewelry': 14,
                'pct_spend_cosmetics': 11,
                'pct_spend_spa': 9,
                'total_spend_3m_percentile': 8,
                'categories_used': 5,
                'is_student': -15,
                'is_premium_client': -20,
                'atm_withdrawal_total_3m': -3,
                'liquidity_need_signal': -8
            },
            'Кредитная карта': {
                'online_services_spend': 18,
                'online_frequency': 12,
                'pct_spend_gaming': 15,
                'pct_spend_streaming': 12,
                'pct_spend_food_delivery': 14,
                'pct_spend_cinema': 10,
                'debt_tx_count': 8,
                'liquidity_need_signal': 10,
                'total_spend_3m_percentile': 7,
                'total_tx_count': 5,
                'is_student': 5,
                'age': -1,
                'debt_to_income_ratio': -12,
                'income_stability_cv': -8
            },
            'Обмен валют': {
                'fx_activity_count': 20,
                'fx_volume_total_3m': 15,
                'fx_tx_count': 12,
                'fx_active_months': 10,
                'main_fx_currency': 8,
                'travel_related_spend': 6,
                'pct_spend_travel': 8,
                'investment_activity_count': 4,
                'avg_monthly_balance_KZT_percentile': 5,
                'total_spend_3m_percentile': 3
            },
            'Кредит наличными': {
                'liquidity_need_signal': 18,
                'atm_withdrawal_count': 12,
                'atm_withdrawal_total_3m': 10,
                'p2p_out_volume': 8,
                'avg_monthly_income': 15,
                'income_stability_cv': -10,
                'is_salary_client': 12,
                'dominant_income_type': 5,
                'debt_to_income_ratio': -15,
                'is_student': -12,
                'savings_potential': -8,
                'avg_monthly_balance_KZT_percentile': -5
            },
            'Депозит Мультивалютный': {
                'fx_activity_count': 15,
                'fx_volume_total_3m': 12,
                'main_fx_currency': 10,
                'avg_monthly_balance_KZT_percentile': 14,
                'savings_potential': 18,
                'balance_to_income_ratio': 11,
                'investment_readiness_score': 8,
                'is_premium_client': 6,
                'liquidity_need_signal': -12,
                'debt_to_income_ratio': -8
            },
            'Депозит Сберегательный': {
                'avg_monthly_balance_KZT_percentile': 20,
                'savings_potential': 22,
                'balance_to_income_ratio': 15,
                'disposable_income': 12,
                'spend_to_income_ratio': -5,
                'liquidity_need_signal': -20,
                'atm_withdrawal_count': -15,
                'debt_to_income_ratio': -10,
                'age': 3,
                'is_premium_client': 8,
                'income_stability_cv': -8
            },
            'Депозит Накопительный': {
                'savings_potential': 20,
                'disposable_income': 15,
                'balance_to_income_ratio': 12,
                'avg_monthly_income': 10,
                'income_stability_cv': -8,
                'is_salary_client': 8,
                'liquidity_need_signal': -8,
                'atm_withdrawal_count': -5,
                'age': 2,
                'is_standard_client': 3
            },
            'Инвестиции': {
                'investment_activity_count': 25,
                'investment_volume': 18,
                'investment_readiness_score': 20,
                'main_investment_type': 5,
                'avg_monthly_balance_KZT_percentile': 10,
                'savings_potential': 12,
                'balance_to_income_ratio': 8,
                'age': -3,
                'is_student': 5,
                'is_premium_client': 8,
                'liquidity_need_signal': -10,
                'debt_to_income_ratio': -8,
                'income_stability_cv': -5
            },
            'Золотые слитки': {
                'avg_monthly_balance_KZT_percentile': 25,
                'savings_potential': 15,
                'balance_to_income_ratio': 18,
                'investment_volume': 12,
                'liquidity_need_signal': -25,
                'atm_withdrawal_count': -20,
                'debt_to_income_ratio': -15,
                'age': 8,
                'is_premium_client': 15,
                'income_stability_cv': -10
            }
        }
        
        self.benefit_params = {
            'Карта для путешествий': {'cashback': 0.04, 'base': 'travel_related_spend'},
            'Премиальная карта': {'cashback': 0.04, 'base': 'luxury_spend'},
            'Кредитная карта': {'cashback': 0.10, 'base': 'online_services_spend'},
            'Депозит Сберегательный': {'rate': 0.165, 'base': 'avg_monthly_balance_KZT'},
            'Депозит Накопительный': {'rate': 0.155, 'base': 'savings_potential'},
            'Депозит Мультивалютный': {'rate': 0.145, 'base': 'avg_monthly_balance_KZT'},
            'Обмен валют': {'saving': 0.005, 'base': 'fx_volume_total_3m'},
            'Инвестиции': {'return': 0.08, 'base': 'savings_potential'},
            'Золотые слитки': {'return': 0.03, 'base': 'avg_monthly_balance_KZT'},
            'Кредит наличными': {'benefit': 0, 'base': 'liquidity_need_signal'}
        }
    
    
    
    def normalize_feature(self, value, feature_name, all_values):
        """Нормализация признаков"""
        if pd.isna(value):
            return 0

        # Обработка категориальных признаков
        if feature_name == 'main_fx_currency':
            return 1 if value not in ['KZT', 0, '0', None] else 0

        # Если значение строка (и это не main_fx_currency) → ставим 0
        if isinstance(value, str):
            return 0

        if '_percentile' in feature_name:
            return value / 100

        if feature_name.startswith('is_') or feature_name.startswith('has_'):
            return float(value)

        if feature_name.startswith('pct_'):
            return min(value / 100, 1.0)

        # Числовые признаки → нормализация по квартилям
        if len(all_values) > 2:
            try:
                numeric_values = pd.to_numeric(all_values, errors='coerce')
                numeric_values = numeric_values[~pd.isna(numeric_values)]
                numeric_values = numeric_values[numeric_values > 0]
                if len(numeric_values) > 0:
                    q75, q25 = np.percentile(numeric_values, [75, 25])
                    iqr = q75 - q25
                if iqr > 0:
                    return max(0, min(1, (value - q25) / iqr))
            except Exception:
                return 0

        return 1.0 if value > 0 else 0.0

    
    def calculate_product_score(self, client_row, product_name, all_features_df):
        """Расчет балла продукта"""
        if product_name not in self.scoring_matrix:
            return 0, {}
        
        score = 0
        weight_sum = 0
        contributing_features = {}
        
        for feature, weight in self.scoring_matrix[product_name].items():
            if feature in client_row.index:
                value = client_row[feature]
                
                all_values = all_features_df[feature].values if feature in all_features_df.columns else [value]
                normalized_value = self.normalize_feature(value, feature, all_values)
                
                contribution = normalized_value * weight
                score += contribution
                weight_sum += abs(weight)
                
                if abs(contribution) > 0.1:
                    contributing_features[feature] = (value, contribution)
        
        if weight_sum > 0:
            score = (score / weight_sum) * 100
        
        return max(0, min(100, score)), contributing_features
    
    def calculate_expected_benefit(self, client_features, product_name):
        """Расчет ожидаемой выгоды"""
        if product_name not in self.benefit_params:
            return 0
        
        params = self.benefit_params[product_name]
        base_value = client_features.get(params['base'], 0)
        
        if 'cashback' in params:
            return base_value * params['cashback']
        elif 'rate' in params:
            return base_value * (params['rate'] / 4)  # Квартальная выгода
        elif 'saving' in params:
            return base_value * params['saving']
        elif 'return' in params:
            return base_value * params['return']
        else:
            return 0
    
    def rank_products(self, client_features, all_features_df):
        """Ранжирование продуктов"""
        results = []
        
        for product_name in self.scoring_matrix.keys():
            score, contributing_features = self.calculate_product_score(client_features, product_name, all_features_df)
            benefit = self.calculate_expected_benefit(client_features, product_name)
            
            # Бизнес-правила
            if product_name == 'Кредит наличными' and client_features.get('avg_monthly_income', 0) < 50000:
                score *= 0.1
            
            if product_name == 'Премиальная карта' and client_features.get('is_student', 0) == 1:
                score *= 0.1
            
            if 'Депозит' in product_name and client_features.get('avg_monthly_balance_KZT', 0) > 1000000:
                score *= 1.3
            
            if product_name == 'Инвестиции' and client_features.get('investment_activity_count', 0) > 0:
                score *= 1.5
            
            results.append({
                'product': product_name,
                'score': score,
                'benefit_kzt': benefit,
                'top_signals': dict(sorted(contributing_features.items(), 
                                          key=lambda x: abs(x[1][1]), 
                                          reverse=True)[:3])
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results


class ContextAwareGenerativeModule:
    """Генератор персонализированных push-уведомлений"""
    
    def __init__(self):
        self.months = {
            1: 'январе', 2: 'феврале', 3: 'марте', 4: 'апреле',
            5: 'мае', 6: 'июне', 7: 'июле', 8: 'августе',
            9: 'сентябре', 10: 'октябре', 11: 'ноябре', 12: 'декабре'
        }
    
    def format_number(self, number):
        """Форматирование чисел по редполитике"""
        if pd.isna(number) or number == 0:
            return "0"
        if number >= 1000:
            return f"{int(number):,}".replace(",", " ")
        return str(int(number))
    
    def get_current_month(self):
        """Текущий месяц для персонализации"""
        return self.months.get(8, 'августе')
    
    def adjust_tone_for_age(self, text, age, status):
        """Корректировка тона для молодежи"""
        if age < 25 or status == 'Студент':
            replacements = [
                ('вы ', 'ты '), ('Вы ', 'Ты '),
                ('ваш', 'твой'), ('Ваш', 'Твой'),
                ('вам ', 'тебе '), ('Вам ', 'Тебе '),
                ('ваши ', 'твои '), ('Ваши ', 'Твои '),
                ('вашу ', 'твою '), ('Вашу ', 'Твою '),
                ('вашей ', 'твоей '), ('Вашей ', 'Твоей ')
            ]
            for old, new in replacements:
                text = text.replace(old, new)
        return text
    
    def get_top_categories(self, row):
        """Извлечение топ-3 категорий трат"""
        category_mapping = {
            'spend_restaurants_3m': 'рестораны',
            'spend_taxi_3m': 'такси',
            'spend_travel_3m': 'путешествия',
            'spend_hotels_3m': 'отели',
            'spend_clothing_3m': 'одежда',
            'spend_groceries_3m': 'продукты',
            'spend_entertainment_3m': 'развлечения',
            'spend_gaming_3m': 'игры',
            'spend_streaming_3m': 'стриминг',
            'spend_food_delivery_3m': 'доставка еды',
            'spend_jewelry_3m': 'ювелирные изделия',
            'spend_cosmetics_3m': 'косметика',
            'spend_spa_3m': 'спа',
            'spend_cinema_3m': 'кино',
            'spend_auto_3m': 'авто',
            'spend_sport_3m': 'спорт'
        }
        
        categories = {}
        for col, name in category_mapping.items():
            if col in row.index:
                value = row.get(col, 0)
                if value > 0:
                    categories[name] = value
        
        top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        return [cat[0] for cat in top_cats]
    
    def generate_travel_card_push(self, row):
        """Генерация пуша для карты путешествий"""
        name = row['name'].split()[0]
        taxi_spend = row.get('spend_taxi_3m', 0)
        taxi_count = row.get('tx_count_taxi_3m', 0)
        travel_spend = row.get('spend_travel_3m', 0)
        hotels_spend = row.get('spend_hotels_3m', 0)
        gas_spend = row.get('spend_gas_3m', 0)
        benefit = row.get('benefit_kzt', 0)
        month = self.get_current_month()
        
        # Высокая активность такси
        if taxi_count > 10 and taxi_spend > 20000:
            return (f"{name}, в {month} вы сделали {int(taxi_count)} поездок на такси на {self.format_number(taxi_spend)} ₸. "
                   f"С картой для путешествий вернули бы ≈{self.format_number(benefit)} ₸. "
                   f"Откройте карту в приложении.")
        
        # Активный путешественник
        elif (travel_spend + hotels_spend) > 50000:
            total_travel = travel_spend + hotels_spend + taxi_spend + gas_spend
            return (f"{name}, ваши траты на поездки составили {self.format_number(total_travel)} ₸. "
                   f"Карта для путешествий вернёт 4% кешбэком. "
                   f"Оформите в приложении.")
        
        # Средняя активность
        elif taxi_spend > 5000:
            return (f"{name}, заметили частые поездки на такси. "
                   f"С тревел-картой часть расходов вернулась бы кешбэком. "
                   f"Хотите оформить?")
        
        # Базовый вариант
        else:
            return (f"{name}, планируете путешествие? "
                   f"Карта для путешествий даст 4% кешбэк на такси и отели. "
                   f"Откройте в приложении.")
    
    def generate_premium_card_push(self, row):
        """Генерация пуша для премиальной карты"""
        name = row['name'].split()[0]
        balance = row.get('avg_monthly_balance_KZT', 0)
        restaurants = row.get('spend_restaurants_3m', 0)
        jewelry = row.get('spend_jewelry_3m', 0)
        cosmetics = row.get('spend_cosmetics_3m', 0)
        spa = row.get('spend_spa_3m', 0)
        total_spend = row.get('total_spend_3m', 0)
        
        # Очень высокий баланс (6+ млн)
        if balance >= 6000000:
            return (f"{name}, у вас высокий остаток на счету это даёт вам большие возможности. "
                   f"Премиальная карта даст до 4% кешбэка на все покупки и бесплатные снятия. "
                   f"Подключите сейчас.")
        
        # Высокий баланс (1-6 млн) + luxury траты
        elif balance >= 1000000 and restaurants > 30000:
            return (f"{name}, у вас стабильно крупный остаток и траты в ресторанах. "
                   f"Премиальная карта даст повышенный кешбэк и бесплатные снятия. "
                   f"Оформить сейчас.")
        
        # Luxury траты
        elif (jewelry + cosmetics + spa) > 50000:
            luxury_total = jewelry + cosmetics + spa
            return (f"{name}, видим покупки премиум-товаров на {self.format_number(luxury_total)} ₸. "
                   f"С премиальной картой вернёте 4% с этих покупок. "
                   f"Подключите карту.")
        
        # Активный транзактор
        elif total_spend > 500000:
            potential_cashback = int(total_spend * 0.03)
            return (f"{name}, за 3 месяца вы потратили {self.format_number(total_spend)} ₸. "
                   f"Премиум карта вернула бы ≈{self.format_number(potential_cashback)} ₸. "
                   f"Оформить сейчас.")
        
        # Стандартный премиум
        else:
            return (f"{name}, ваш финансовый профиль соответствует премиальному сегменту. "
                   f"Получайте до 4% кешбэка и привилегии Visa Signature. "
                   f"Оформить сейчас.")
    
    def generate_credit_card_push(self, row):
        """Генерация пуша для кредитной карты"""
        name = row['name'].split()[0]
        online_spend = row.get('online_services_spend', 0)
        gaming = row.get('spend_gaming_3m', 0)
        streaming = row.get('spend_streaming_3m', 0)
        delivery = row.get('spend_food_delivery_3m', 0)
        cinema = row.get('spend_cinema_3m', 0)
        top_categories = self.get_top_categories(row)
        
        # Активный онлайн-пользователь
        if online_spend > 30000:
            online_cats = []
            if gaming > 0: online_cats.append('игры')
            if delivery > 0: online_cats.append('доставка')
            if cinema > 0: online_cats.append('кино')
            if streaming > 0: online_cats.append('стриминг')
            
            if len(online_cats) >= 2:
                cats_str = ', '.join(online_cats[:3])
                return (f"{name}, ваши топ-категории — {cats_str}. "
                       f"Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. "
                       f"Оформить карту.")
        
        # Есть топ категории
        if len(top_categories) >= 3:
            cats_str = ', '.join(top_categories[:3])
            return (f"{name}, ваши топ-категории — {cats_str}. "
                   f"Кредитная карта даёт до 10% в любимых категориях. "
                   f"Оформить карту.")
        elif len(top_categories) >= 2:
            cats_str = ', '.join(top_categories[:2])
            return (f"{name}, часто покупаете в категориях {cats_str}. "
                   f"Получайте до 10% кешбэка с кредитной картой. "
                   f"Оформить карту.")
        
        # Базовый вариант
        return (f"{name}, получайте до 10% кешбэка в трёх любимых категориях. "
               f"2 месяца без процентов на покупки. "
               f"Оформить карту.")
    
    def generate_fx_push(self, row):
        """Генерация пуша для обмена валют"""
        name = row['name'].split()[0]
        fx_volume = row.get('fx_volume_total_3m', 0)
        fx_count = row.get('fx_activity_count', 0)
        main_currency = row.get('main_fx_currency', 'USD')
        
        # Активный трейдер
        if fx_count > 5 and fx_volume > 500000:
            return (f"{name}, вы часто меняете валюту — {int(fx_count)} операций за 3 месяца. "
                   f"В приложении выгодный курс и авто-покупка по целевому курсу. "
                   f"Настроить обмен.")
        
        # Есть основная валюта
        elif main_currency not in ['KZT', 0, '0', None] and fx_volume > 100000:
            curr_map = {'USD': 'долларах', 'EUR': 'евро', 'RUB': 'рублях'}
            curr_name = curr_map.get(str(main_currency), 'валюте')
            return (f"{name}, вы часто платите в {curr_name}. "
                   f"В приложении выгодный обмен и авто-покупка по целевому курсу. "
                   f"Настроить обмен.")
        
        # Периодический обмен
        elif fx_count > 0:
            return (f"{name}, меняйте валюту выгоднее в приложении. "
                   f"Без комиссии, 24/7, моментальные операции. "
                   f"Попробовать сейчас.")
        
        # Базовый
        else:
            return (f"{name}, попробуйте обмен валют в приложении. "
                   f"Выгодный курс без комиссии 24/7. "
                   f"Настроить обмен.")
    
    def generate_credit_cash_push(self, row):
        """Генерация пуша для кредита наличными"""
        name = row['name'].split()[0]
        income = row.get('avg_monthly_income', 0)
        liquidity_signal = row.get('liquidity_need_signal', 0)
        atm_count = row.get('atm_withdrawal_count', 0)
        
        # Есть доход и потребность в ликвидности
        if income > 100000 and liquidity_signal > 5:
            return (f"{name}, если нужны средства на крупные покупки — оформите кредит наличными. "
                   f"Гибкие условия, досрочное погашение без штрафов. "
                   f"Узнать доступный лимит.")
        
        # Стабильный доход
        elif income > 200000:
            return (f"{name}, вам доступен кредит наличными до 2 000 000 ₸. "
                   f"Без залога и справок, решение за 15 минут. "
                   f"Узнать лимит.")
        
        # Частые снятия наличных
        elif atm_count > 10:
            return (f"{name}, часто снимаете наличные? Кредит даст больше возможностей. "
                   f"Быстрое оформление, гибкие условия. "
                   f"Узнать доступный лимит.")
        
        # Базовый
        else:
            return (f"{name}, нужны средства? Кредит наличными без залога и справок. "
                   f"Оформление онлайн, гибкие условия. "
                   f"Узнать доступный лимит.")
    
    def generate_deposit_push(self, row, deposit_type):
        """Генерация пуша для депозитов"""
        name = row['name'].split()[0]
        balance = row.get('avg_monthly_balance_KZT', 0)
        savings = row.get('savings_potential', 0)
        disposable = row.get('disposable_income', 0)
        
        rates = {
            'Депозит Мультивалютный': '14,50%',
            'Депозит Сберегательный': '16,50%',
            'Депозит Накопительный': '15,50%'
        }
        
        if deposit_type == 'Депозит Сберегательный':
            if balance > 5000000:
                monthly_income = int(balance * 0.165 / 12)
                return (f"{name}, у вас остаются свободные средства. "
                       f"Разместите их на вкладе — удобно копить и получать вознаграждение. "
                       f"Открыть вклад.")
            elif savings > 100000:
                return (f"{name}, ваши накопления могут приносить {rates[deposit_type]} годовых. "
                       f"Максимальная ставка для надёжного сохранения. "
                       f"Открыть вклад.")
            else:
                return (f"{name}, начните копить под {rates[deposit_type]} годовых. "
                       f"Защита от инфляции и гарантия KDIF. "
                       f"Открыть вклад.")
        
        elif deposit_type == 'Депозит Накопительный':
            if disposable > 50000:
                return (f"{name}, откладывайте {self.format_number(disposable)} ₸ ежемесячно под {rates[deposit_type]}. "
                       f"Пополняйте когда удобно, растите капитал. "
                       f"Открыть вклад.")
            else:
                return (f"{name}, начните откладывать регулярно под {rates[deposit_type]} годовых. "
                       f"Удобное пополнение, надёжное накопление. "
                       f"Открыть вклад.")
        
        else:  # Мультивалютный
            fx_activity = row.get('fx_activity_count', 0)
            if fx_activity > 0:
                return (f"{name}, храните сбережения в разных валютах под {rates[deposit_type]}. "
                       f"Свободный доступ к средствам в любой момент. "
                       f"Открыть вклад.")
            else:
                return (f"{name}, защитите накопления от курсовых рисков. "
                       f"Мультивалютный вклад под {rates[deposit_type]} с доступом 24/7. "
                       f"Открыть вклад.")
    
    def generate_investment_push(self, row):
        """Генерация пуша для инвестиций"""
        name = row['name'].split()[0]
        status = row.get('status', '')
        balance = row.get('avg_monthly_balance_KZT', 0)
        investment_activity = row.get('investment_activity_count', 0)
        age = row.get('age', 30)
        
        # Уже инвестирует
        if investment_activity > 0:
            return (f"{name}, расширьте инвестпортфель без комиссий в первый год. "
                   f"Доступ к мировым рынкам из приложения. "
                   f"Открыть счёт.")
        
        # Премиальный клиент
        elif status == 'Премиальный клиент':
            return (f"{name}, диверсифицируйте капитал через инвестиции. "
                   f"Особые условия для премиальных клиентов. "
                   f"Начать инвестировать.")
        
        # Высокий баланс
        elif balance > 1000000:
            return (f"{name}, ваши средства могут работать эффективнее. "
                   f"Инвестиции от 6 ₸, без комиссий на старте. "
                   f"Открыть счёт.")
        
        # Молодой клиент
        elif age < 30:
            return (f"{name}, начните инвестировать с малых сумм. "
                   f"Порог входа от 6 ₸, обучение в приложении. "
                   f"Попробовать сейчас.")
        
        # Базовый
        else:
            return (f"{name}, попробуйте инвестиции с низким порогом входа. "
                   f"0% комиссий в первый год. "
                   f"Открыть счёт.")
    
    def generate_gold_push(self, row):
        """Генерация пуша для золотых слитков"""
        name = row['name'].split()[0]
        balance = row.get('avg_monthly_balance_KZT', 0)
        age = row.get('age', 30)
        status = row.get('status', '')
        
        # Очень высокий баланс
        if balance > 10000000:
            return (f"{name}, диверсифицируйте активы золотыми слитками. "
                   f"999,9 проба, хранение в сейфовых ячейках банка. "
                   f"Узнать подробности.")
        
        # Возрастной клиент с накоплениями
        elif age > 45 and balance > 3000000:
            return (f"{name}, сохраните капитал для будущего в золоте. "
                   f"Защита от инфляции, физический актив. "
                   f"Заказать консультацию.")
        
        # Премиальный статус
        elif status == 'Премиальный клиент':
            return (f"{name}, добавьте золото в портфель активов. "
                   f"Слитки 999,9 пробы, особые условия хранения. "
                   f"Узнать подробности.")
        
        # Стандартный
        else:
            return (f"{name}, золотые слитки — надёжное сохранение стоимости. "
                   f"Доступны разные веса, предзаказ в приложении. "
                   f"Узнать подробности.")
    
    def validate_push_message(self, message):
        """Валидация push-уведомления согласно редполитике"""
        # Проверка длины (180-220 символов оптимально)
        if len(message) > 250:
            sentences = message.split('. ')
            if len(sentences) > 2:
                message = '. '.join(sentences[:2]) + '.'
        
        # Проверка на КАПС
        if message.isupper():
            message = message.capitalize()
        
        # Проверка на множественные восклицательные знаки
        message = re.sub(r'!+', '!', message)
        if message.count('!') > 1:
            message = message.replace('!', '.', message.count('!') - 1)
        
        # Убираем лишние пробелы
        message = ' '.join(message.split())
        
        # Убираем двойные запятые или точки
        message = re.sub(r'[,]+', ',', message)
        message = re.sub(r'[.]+', '.', message)
        
        return message
    
    def generate_push(self, row):
        """Главная функция генерации push-уведомления"""
        product = row['product']
        
        # Маппинг генераторов
        generators = {
            'Карта для путешествий': self.generate_travel_card_push,
            'Премиальная карта': self.generate_premium_card_push,
            'Кредитная карта': self.generate_credit_card_push,
            'Обмен валют': self.generate_fx_push,
            'Кредит наличными': self.generate_credit_cash_push,
            'Депозит Мультивалютный': lambda r: self.generate_deposit_push(r, 'Депозит Мультивалютный'),
            'Депозит Сберегательный': lambda r: self.generate_deposit_push(r, 'Депозит Сберегательный'),
            'Депозит Накопительный': lambda r: self.generate_deposit_push(r, 'Депозит Накопительный'),
            'Инвестиции': self.generate_investment_push,
            'Золотые слитки': self.generate_gold_push
        }
        
        # Генерация сообщения
        if product in generators:
            message = generators[product](row)
        else:
            name = row['name'].split()[0]
            message = f"{name}, у нас есть специальное предложение для вас. Узнать подробности в приложении."
        
        # Корректировка тона для молодёжи
        age = row.get('age', 30)
        status = row.get('status', '')
        message = self.adjust_tone_for_age(message, age, status)
        
        # Валидация согласно редполитике
        message = self.validate_push_message(message)
        
        return message


def run_helios_pipeline(data_folder):
    """Запуск полного пайплайна HELIOS"""
    
    print("\n" + "="*60)
    print("🚀 HELIOS MASTER PIPELINE v2.0")
    print("="*60)
    print("Архитектура: CFS → QRE → CGM")
    print("="*60 + "\n")
    
    # ЭТАП 1: Client Feature Store
    print("\n📊 ЭТАП 1: Создание хранилища признаков (CFS)")
    print("-" * 60)
    
    cfs = ClientFeatureStore()
    features_df = cfs.create_advanced_features(data_folder)
    
    # Сохранение features
    features_file = os.path.join(data_folder, 'advanced_client_features.csv')
    features_df.to_csv(features_file, index=False, encoding='utf-8-sig')
    print(f"✅ Сохранено в {features_file}")
    
    # ЭТАП 2: Quantitative Recommendation Engine
    print("\n🎯 ЭТАП 2: Генерация рекомендаций (QRE)")
    print("-" * 60)
    
    qre = QuantitativeRecommendationEngine()
    recommendations = []
    
    for idx, row in features_df.iterrows():
        client_code = row['client_code']
        client_name = row['name']
        
        # Ранжирование продуктов
        ranked_products = qre.rank_products(row, features_df)
        
        # Выбор лучшего продукта
        best_product = ranked_products[0]
        
        # Сбор данных для генерации пушей
        rec_data = {
            'client_code': client_code,
            'name': client_name,
            'status': row['status'],
            'age': row['age'],
            'product': best_product['product'],
            'score': best_product['score'],
            'benefit_kzt': best_product['benefit_kzt'],
            'top4_products': [p['product'] for p in ranked_products[:4]]
        }
        
        # Добавление релевантных признаков для генерации
        relevant_features = [
            'spend_taxi_3m', 'tx_count_taxi_3m', 'spend_travel_3m', 'spend_hotels_3m',
            'spend_restaurants_3m', 'spend_jewelry_3m', 'spend_cosmetics_3m', 'spend_spa_3m',
            'online_services_spend', 'spend_gaming_3m', 'spend_streaming_3m', 'spend_food_delivery_3m',
            'spend_cinema_3m', 'fx_volume_total_3m', 'fx_activity_count', 'main_fx_currency',
            'avg_monthly_balance_KZT', 'total_spend_3m', 'savings_potential', 'disposable_income',
            'avg_monthly_income', 'liquidity_need_signal', 'investment_activity_count',
            'atm_withdrawal_count', 'spend_gas_3m'
        ]
        
        for feature in relevant_features:
            if feature in row.index:
                rec_data[feature] = row[feature]
            else:
                rec_data[feature] = 0
        
        recommendations.append(rec_data)
        
        # Примеры для первых клиентов
        if idx < 3:
            print(f"\n👤 Клиент {client_code} ({client_name}):")
            print(f"   ✅ Продукт: {best_product['product']}")
            print(f"   📊 Балл: {best_product['score']:.2f}")
            print(f"   💰 Выгода: {best_product['benefit_kzt']:,.0f} ₸")
    
    recommendations_df = pd.DataFrame(recommendations)
    qre_file = os.path.join(data_folder, 'qre_recommendations.csv')
    recommendations_df.to_csv(qre_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Рекомендации сохранены в {qre_file}")
    
    # ЭТАП 3: Context-Aware Generative Module
    print("\n💬 ЭТАП 3: Генерация push-уведомлений (CGM)")
    print("-" * 60)
    
    cgm = ContextAwareGenerativeModule()
    push_notifications = []
    
    for idx, row in recommendations_df.iterrows():
        push_message = cgm.generate_push(row)
        push_notifications.append(push_message)
        
        # Примеры для первых клиентов
        if idx < 3:
            print(f"\n👤 Клиент {row['client_code']} ({row['name']}):")
            print(f"   📱 Push: {push_message}")
            print(f"   📏 Длина: {len(push_message)} символов")
    
    # Создание финального DataFrame
    final_df = pd.DataFrame({
        'client_code': recommendations_df['client_code'],
        'product': recommendations_df['product'],
        'push_notification': push_notifications
    })
    
    # Сохранение результата
    output_file = os.path.join(data_folder, 'submission.csv')
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Запуск HELIOS Pipeline")
    parser.add_argument(
        "-d", "--data_folder",
        help="Путь к папке с данными (clients.csv, transactions и transfers)."
    )
    args = parser.parse_args()

    default_folder = Path(__file__).resolve().parent
    data_folder = Path(
        args.data_folder
        or os.getenv("DATA_FOLDER")
        or default_folder
    )

    if data_folder.is_dir():
        run_helios_pipeline(str(data_folder))
        print("\n🎯 ГОТОВО! Результат сохранён в submission.csv")
    else:
        print(f"❌ Папка '{data_folder}' не найдена")
        sys.exit(1)