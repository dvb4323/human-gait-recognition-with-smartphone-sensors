/**
 * Confidence Bars Component - Shows probability for each activity
 */

import React from 'react';
import {View, Text, StyleSheet} from 'react-native';
import {ACTIVITY_NAMES, ACTIVITY_COLORS} from '../utils/constants';

interface ConfidenceBarsProps {
  probabilities: number[];
}

const ConfidenceBars: React.FC<ConfidenceBarsProps> = ({probabilities}) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Confidence Breakdown</Text>
      {probabilities.map((prob, index) => (
        <View key={index} style={styles.barContainer}>
          <Text style={styles.label}>{ACTIVITY_NAMES[index]}</Text>
          <View style={styles.barBackground}>
            <View
              style={[
                styles.barFill,
                {
                  width: `${prob * 100}%`,
                  backgroundColor: ACTIVITY_COLORS[index],
                },
              ]}
            />
          </View>
          <Text style={styles.percentage}>{(prob * 100).toFixed(0)}%</Text>
        </View>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#16213e',
    borderRadius: 15,
    padding: 20,
    marginVertical: 10,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 15,
  },
  barContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  label: {
    width: 100,
    fontSize: 14,
    color: '#b0b0b0',
  },
  barBackground: {
    flex: 1,
    height: 20,
    backgroundColor: '#0f1419',
    borderRadius: 10,
    overflow: 'hidden',
    marginHorizontal: 10,
  },
  barFill: {
    height: '100%',
    borderRadius: 10,
  },
  percentage: {
    width: 45,
    fontSize: 14,
    color: '#ffffff',
    textAlign: 'right',
  },
});

export default ConfidenceBars;
