/**
 * Activity Display Component - Shows current predicted activity
 */

import React from 'react';
import {View, Text, StyleSheet} from 'react-native';
import {ACTIVITY_COLORS, ACTIVITY_ICONS} from '../utils/constants';

interface ActivityDisplayProps {
  activity: string;
  confidence: number;
  activityIndex: number;
}

const ActivityDisplay: React.FC<ActivityDisplayProps> = ({
  activity,
  confidence,
  activityIndex,
}) => {
  const color = ACTIVITY_COLORS[activityIndex] || '#4CAF50';
  const icon = ACTIVITY_ICONS[activityIndex] || '🚶';

  return (
    <View style={[styles.container, {borderColor: color}]}>
      <Text style={styles.icon}>{icon}</Text>
      <Text style={styles.activity}>{activity}</Text>
      <Text style={[styles.confidence, {color}]}>
        {(confidence * 100).toFixed(1)}%
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#16213e',
    borderRadius: 20,
    borderWidth: 3,
    padding: 30,
    alignItems: 'center',
    marginVertical: 20,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 4},
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  icon: {
    fontSize: 64,
    marginBottom: 10,
  },
  activity: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  confidence: {
    fontSize: 36,
    fontWeight: '900',
  },
});

export default ActivityDisplay;
