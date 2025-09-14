import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import json

class RuleBasedLevelClassifier:
    def __init__(self):
        self.rules = self._define_rules()
        
    def _define_rules(self):
        """Define clear rules for each learning level"""
        return {
            'beginner': {
                'keywords': ['beginner', 'basic', 'intro', 'introduction', 'fundamental', 
                           'starter', 'learn', 'tutorial', 'easy', 'simple', 'basics',
                           'getting started', 'crash course', 'overview', 'foundation'],
                'duration_max': 25,  # minutes
                'keyword_count_max': 8,
                'title_indicators': ['for beginners', 'beginners guide', 'learn from scratch'],
                'engagement_min': 0.001
            },
            'advanced': {
                'keywords': ['advanced', 'expert', 'master', 'deep dive', 'complex', 
                           'professional', 'production', 'optimization', 'architecture',
                           'design patterns', 'scalability', 'performance', 'enterprise',
                           'masterclass', 'advanced techniques', 'expert level'],
                'duration_min': 30,  # minutes
                'keyword_count_min': 10,
                'title_indicators': ['advanced tutorial', 'expert guide', 'master class'],
                'engagement_max': 0.003  # Advanced content often has lower engagement
            },
            'intermediate': {
                'keywords': ['intermediate', 'guide', 'course', 'walkthrough', 'explained',
                           'understanding', 'concepts', 'techniques', 'methods', 'comprehensive',
                           'complete guide', 'in-depth', 'detailed', 'step by step'],
                'duration_range': (15, 45),
                'keyword_count_range': (6, 12),
                # Intermediate is the default, so fewer specific indicators
            }
        }
    
    def load_and_preprocess(self, file_path):
        """Load and preprocess the dataset"""
        df = pd.read_csv(file_path)
        
        print(f"📥 Original dataset size: {len(df)} videos")
        
        # Clean data
        df = df.dropna(subset=['keywords', 'title', 'description'])
        df = df[df['keywords'] != 'No keywords']
        df = df[df['keyword_count'] > 2]
        
        print(f"📊 After cleaning: {len(df)} videos")
        
        # Create features
        df['title_lower'] = df['title'].str.lower()
        df['description_lower'] = df['description'].str.lower()
        df['engagement_ratio'] = np.log1p(df['likes'] / (df['views'].replace(0, 1) + 1))
        
        return df
    
    def classify_video(self, row):
        """Classify a single video using rule-based approach"""
        title = row['title_lower']
        keywords = row['keywords'].lower()
        duration = row['duration_minutes']
        keyword_count = row['keyword_count']
        engagement = row.get('engagement_ratio', 0)
        
        # Calculate scores for each level
        scores = {
            'beginner': self._calculate_beginner_score(title, keywords, duration, keyword_count, engagement),
            'intermediate': self._calculate_intermediate_score(title, keywords, duration, keyword_count),
            'advanced': self._calculate_advanced_score(title, keywords, duration, keyword_count, engagement)
        }
        
        # Return the level with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_beginner_score(self, title, keywords, duration, keyword_count, engagement):
        """Calculate beginner score"""
        score = 0
        rules = self.rules['beginner']
        
        # Keyword matches
        for keyword in rules['keywords']:
            if keyword in keywords or keyword in title:
                score += 3
        
        # Title indicators
        for indicator in rules['title_indicators']:
            if indicator in title:
                score += 4
        
        # Duration (shorter is better for beginner)
        if duration <= rules['duration_max']:
            score += (rules['duration_max'] - duration) / 5
        
        # Keyword count (fewer is better for beginner)
        if keyword_count <= rules['keyword_count_max']:
            score += (rules['keyword_count_max'] - keyword_count) / 2
        
        # Engagement (higher is better for beginner)
        if engagement >= rules['engagement_min']:
            score += engagement * 1000
        
        return score
    
    def _calculate_advanced_score(self, title, keywords, duration, keyword_count, engagement):
        """Calculate advanced score"""
        score = 0
        rules = self.rules['advanced']
        
        # Keyword matches
        for keyword in rules['keywords']:
            if keyword in keywords or keyword in title:
                score += 3
        
        # Title indicators
        for indicator in rules['title_indicators']:
            if indicator in title:
                score += 4
        
        # Duration (longer is better for advanced)
        if duration >= rules['duration_min']:
            score += (duration - rules['duration_min']) / 10
        
        # Keyword count (more is better for advanced)
        if keyword_count >= rules['keyword_count_min']:
            score += (keyword_count - rules['keyword_count_min']) / 2
        
        # Engagement (moderate for advanced)
        if engagement <= rules['engagement_max']:
            score += 2
        
        return score
    
    def _calculate_intermediate_score(self, title, keywords, duration, keyword_count):
        """Calculate intermediate score"""
        score = 2  # Base score for intermediate (default)
        rules = self.rules['intermediate']
        
        # Keyword matches
        for keyword in rules['keywords']:
            if keyword in keywords or keyword in title:
                score += 2
        
        # Duration in intermediate range
        if rules['duration_range'][0] <= duration <= rules['duration_range'][1]:
            duration_center = (rules['duration_range'][0] + rules['duration_range'][1]) / 2
            score += 3 - abs(duration - duration_center) / 10
        
        # Keyword count in intermediate range
        if rules['keyword_count_range'][0] <= keyword_count <= rules['keyword_count_range'][1]:
            keyword_center = (rules['keyword_count_range'][0] + rules['keyword_count_range'][1]) / 2
            score += 2 - abs(keyword_count - keyword_center) / 4
        
        return score
    
    def classify_dataset(self, df):
        """Classify all videos in the dataset"""
        print("🔍 Classifying videos using rule-based approach...")
        
        results = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"   Processed {idx}/{len(df)} videos...")
            
            level = self.classify_video(row)
            results.append(level)
        
        df['learning_level'] = results
        return df
    
    def analyze_distribution(self, df):
        """Analyze and balance the distribution if needed"""
        level_counts = df['learning_level'].value_counts()
        total_videos = len(df)
        
        print(f"\n📊 Initial Distribution:")
        for level in ['beginner', 'intermediate', 'advanced']:
            count = level_counts.get(level, 0)
            percentage = (count / total_videos) * 100
            print(f"   {level.capitalize()}: {count} videos ({percentage:.1f}%)")
        
        # If distribution is unbalanced, adjust
        if level_counts.get('intermediate', 0) / total_videos < 0.25:
            print("⚠️ Intermediate videos underrepresented, adjusting...")
            df = self._balance_distribution(df)
        
        return df
    
    def _balance_distribution(self, df):
        """Balance the distribution by reclassifying edge cases"""
        # Get videos that are borderline between levels
        borderline_videos = []
        
        for idx, row in df.iterrows():
            title = row['title_lower']
            keywords = row['keywords'].lower()
            duration = row['duration_minutes']
            keyword_count = row['keyword_count']
            
            # Calculate all scores
            b_score = self._calculate_beginner_score(title, keywords, duration, keyword_count, 0)
            i_score = self._calculate_intermediate_score(title, keywords, duration, keyword_count)
            a_score = self._calculate_advanced_score(title, keywords, duration, keyword_count, 0)
            
            scores = [b_score, i_score, a_score]
            max_score = max(scores)
            second_score = sorted(scores)[-2]
            
            # If scores are close, it's a borderline case
            if max_score - second_score < 2:
                borderline_videos.append((idx, scores))
        
        # Reclassify some borderline cases to intermediate
        intermediate_target = int(len(df) * 0.35)  # Target 35% intermediate
        current_intermediate = len(df[df['learning_level'] == 'intermediate'])
        needed = max(0, intermediate_target - current_intermediate)
        
        print(f"   Need to reclassify {needed} videos to intermediate")
        
        # Sort borderline videos by how close they are to intermediate
        borderline_sorted = sorted(borderline_videos, 
                                 key=lambda x: abs(x[1][1] - max(x[1][0], x[1][2])))
        
        for idx, scores in borderline_sorted[:needed]:
            df.at[idx, 'learning_level'] = 'intermediate'
        
        return df
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations"""
        sns.set_style("whitegrid")
        plt.figure(figsize=(18, 12))
        
        # 1. Learning level distribution
        plt.subplot(2, 3, 1)
        level_counts = df['learning_level'].value_counts()
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Beginner, Intermediate, Advanced
        plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Learning Level Distribution')
        
        # 2. Duration by learning level
        plt.subplot(2, 3, 2)
        sns.boxplot(data=df, x='learning_level', y='duration_minutes', 
                   order=['beginner', 'intermediate', 'advanced'], palette=colors)
        plt.title('Duration by Learning Level')
        plt.xlabel('Learning Level')
        plt.ylabel('Duration (minutes)')
        
        # 3. Keyword count by learning level
        plt.subplot(2, 3, 3)
        sns.violinplot(data=df, x='learning_level', y='keyword_count', 
                      order=['beginner', 'intermediate', 'advanced'], palette=colors)
        plt.title('Keyword Count by Learning Level')
        plt.xlabel('Learning Level')
        plt.ylabel('Keyword Count')
        
        # 4. Views by learning level
        plt.subplot(2, 3, 4)
        sns.barplot(data=df, x='learning_level', y='views', estimator=np.median,
                   order=['beginner', 'intermediate', 'advanced'], palette=colors, ci=None)
        plt.title('Median Views by Learning Level')
        plt.xlabel('Learning Level')
        plt.ylabel('Views (log scale)')
        plt.yscale('log')
        
        # 5. Engagement by learning level
        plt.subplot(2, 3, 5)
        if 'engagement_ratio' in df.columns:
            sns.barplot(data=df, x='learning_level', y='engagement_ratio', estimator=np.mean,
                       order=['beginner', 'intermediate', 'advanced'], palette=colors, ci='sd')
            plt.title('Engagement Ratio by Learning Level')
            plt.xlabel('Learning Level')
            plt.ylabel('Engagement Ratio')
        
        # 6. Duration vs Keywords scatter
        plt.subplot(2, 3, 6)
        scatter = sns.scatterplot(data=df, x='duration_minutes', y='keyword_count', 
                                hue='learning_level', hue_order=['beginner', 'intermediate', 'advanced'],
                                palette=colors, alpha=0.7, s=50)
        plt.title('Duration vs Keyword Count')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Keyword Count')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('rule_based_classification.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_enhanced_dataset(self, df):
        """Create enhanced dataset"""
        enhanced_df = df.copy()
        
        # Add binary flags
        enhanced_df['is_beginner'] = (enhanced_df['learning_level'] == 'beginner').astype(int)
        enhanced_df['is_intermediate'] = (enhanced_df['learning_level'] == 'intermediate').astype(int)
        enhanced_df['is_advanced'] = (enhanced_df['learning_level'] == 'advanced').astype(int)
        
        # Capitalize learning levels
        enhanced_df['learning_level'] = enhanced_df['learning_level'].str.capitalize()
        
        # Show final distribution
        level_counts = enhanced_df['learning_level'].value_counts()
        print(f"\n🎯 Final Learning Level Distribution:")
        for level in ['Beginner', 'Intermediate', 'Advanced']:
            count = level_counts.get(level, 0)
            percentage = (count / len(enhanced_df)) * 100
            print(f"   {level}: {count} videos ({percentage:.1f}%)")
        
        return enhanced_df

# Main execution
def main():
    print("🚀 Starting rule-based classification for 590 videos...")
    classifier = RuleBasedLevelClassifier()
    
    # Load data
    df = classifier.load_and_preprocess('fast_dataset.csv')
    
    # Classify videos
    df_classified = classifier.classify_dataset(df)
    
    # Analyze and balance distribution
    df_balanced = classifier.analyze_distribution(df_classified)
    
    # Create visualizations
    classifier.create_visualizations(df_balanced)
    
    # Create enhanced dataset
    enhanced_df = classifier.create_enhanced_dataset(df_balanced)
    
    # Save results
    enhanced_df.to_csv('rule_based_classified_dataset.csv', index=False)
    
    print(f"\n🎉 Rule-based classification completed!")
    print(f"💾 Dataset saved to 'rule_based_classified_dataset.csv'")
    print(f"📈 Visualizations saved as 'rule_based_classification.png'")

if __name__ == "__main__":
    main()