import numpy as np


def _get_new_strategy_params(sensitivity_level):
    if sensitivity_level == "灵敏":
        return {
            'time_window_ms': 2500,
            'height_threshold': 0.4,
            'velocity_threshold': 0.4,
            'acceleration_threshold': 1.5,
            'low_height_threshold': 0.4,
            'min_height_low': 0.6,
            'low_duration_threshold': 0.3,
            'min_conditions': 5,
            'max_height_min': -2.0,
            'max_height_max': 1.8,
            'slow_velocity_threshold': 0.25,
        }
    if sensitivity_level == "不灵敏":
        return {
            'time_window_ms': 3500,
            'height_threshold': 0.6,
            'velocity_threshold': 0.6,
            'acceleration_threshold': 2.5,
            'low_height_threshold': 0.2,
            'min_height_low': 0.4,
            'low_duration_threshold': 0.7,
            'min_conditions': 7,
            'max_height_min': -2.0,
            'max_height_max': 1.2,
            'slow_velocity_threshold': 0.35,
        }
    return {
        'time_window_ms': 3000,
        'height_threshold': 0.5,
        'velocity_threshold': 0.5,
        'acceleration_threshold': 2.0,
        'low_height_threshold': 0.3,
        'min_height_low': 0.5,
        'low_duration_threshold': 0.5,
        'min_conditions': 6,
        'max_height_min': -2.0,
        'max_height_max': 1.5,
        'slow_velocity_threshold': 0.3,
    }


def _is_height_outlier(new_z, new_y, new_range, history, max_height_range=2.0, max_y_range=2.0):
    if len(history) == 0:
        return abs(new_z) > max_height_range or abs(new_y) > max_y_range

    if abs(new_z) > max_height_range or abs(new_y) > max_y_range:
        return True

    last_z = history[-1][1]
    last_y = history[-1][2]
    last_range = history[-1][3]
    height_change = abs(new_z - last_z)
    y_change = abs(new_y - last_y)
    range_change = abs(new_range - last_range)

    if range_change < 0.1:
        if new_range < 0.5:
            threshold_z, threshold_y = 0.15, 0.2
        elif new_range < 1.0:
            threshold_z, threshold_y = 0.25, 0.3
        else:
            threshold_z, threshold_y = 0.3, 0.4
        if height_change > threshold_z or y_change > threshold_y:
            return True

    if new_range < 1.0:
        if new_range < 0.5:
            max_change_z, max_change_y = 0.3, 0.4
        else:
            max_change_z, max_change_y = 0.4, 0.5
        if height_change > max_change_z or y_change > max_change_y:
            return True

    if len(history) >= 5:
        recent_heights = np.array([point[1] for point in history[-10:]], dtype=np.float64)
        recent_y_values = np.array([point[2] for point in history[-10:]], dtype=np.float64)

        q1_z = np.percentile(recent_heights, 25)
        q3_z = np.percentile(recent_heights, 75)
        iqr_z = q3_z - q1_z
        q1_y = np.percentile(recent_y_values, 25)
        q3_y = np.percentile(recent_y_values, 75)
        iqr_y = q3_y - q1_y

        if iqr_z < 0.1:
            median_height = np.median(recent_heights)
            if new_range < 0.5:
                threshold = 0.2
            elif new_range < 1.0:
                threshold = 0.3
            else:
                threshold = 0.5
            if abs(new_z - median_height) > threshold:
                return True
        else:
            if new_range < 0.5:
                multiplier = 1.0
            elif new_range < 1.0:
                multiplier = 1.5
            else:
                multiplier = 2.0
            lower_bound_z = q1_z - multiplier * iqr_z
            upper_bound_z = q3_z + multiplier * iqr_z
            if new_z < lower_bound_z or new_z > upper_bound_z:
                return True

        if iqr_y < 0.1:
            median_y = np.median(recent_y_values)
            if new_range < 0.5:
                threshold_y = 0.3
            elif new_range < 1.0:
                threshold_y = 0.4
            else:
                threshold_y = 0.5
            if abs(new_y - median_y) > threshold_y:
                return True
        else:
            if new_range < 0.5:
                multiplier_y = 1.0
            elif new_range < 1.0:
                multiplier_y = 1.2
            else:
                multiplier_y = 1.5
            lower_bound_y = q1_y - multiplier_y * iqr_y
            upper_bound_y = q3_y + multiplier_y * iqr_y
            if new_y < lower_bound_y or new_y > upper_bound_y:
                return True

    return False


def update_height_history(history, sample, max_history):
    new_history = list(history)
    timestamp_ms, z_value, y_value, range_value = sample

    if not _is_height_outlier(z_value, y_value, range_value, new_history):
        new_history.append(sample)
    elif len(new_history) >= 2:
        recent_z = [point[1] for point in new_history[-3:]]
        recent_y = [point[2] for point in new_history[-3:]]
        smoothed_z = float(np.median(recent_z))
        smoothed_y = float(np.median(recent_y))
        if abs(smoothed_z) < 2.0 and abs(smoothed_y) < 2.0:
            new_history.append((timestamp_ms, smoothed_z, smoothed_y, range_value))

    if len(new_history) > max_history:
        new_history = new_history[-max_history:]
    return new_history


def _collect_recent_points(history, time_window_ms, current_time_ms):
    return [point for point in history if current_time_ms - point[0] <= time_window_ms]


def _empty_result(strategy=None):
    return {
        'detected': False,
        'strategy': strategy,
        'message': '',
        'metrics': {},
    }


def detect_fall(history, settings, current_time_ms):
    """新策略唯一跌倒检测入口。

    Args:
        history: height_history 列表，格式 [(timestamp_ms, z, y, range), ...]
        settings: 字典，必含 'sensitivity_level'（'灵敏'/'中等'/'不灵敏'）
        current_time_ms: 当前时间戳（毫秒）

    Returns:
        dict，含 detected、strategy、message、metrics 字段。
    """
    if len(history) < 2:
        return _empty_result(strategy='new')

    sensitivity_level = settings.get('sensitivity_level', '中等')
    params = _get_new_strategy_params(sensitivity_level)
    recent_points = _collect_recent_points(history, params['time_window_ms'], current_time_ms)
    if len(recent_points) < 2:
        return _empty_result(strategy='new')

    recent_heights = [point[1] for point in recent_points]
    recent_y_values = [point[2] for point in recent_points]
    recent_timestamps = [point[0] for point in recent_points]
    max_height_idx = int(np.argmax(recent_heights))
    min_height_idx = int(np.argmin(recent_heights))

    if recent_timestamps[min_height_idx] <= recent_timestamps[max_height_idx]:
        return _empty_result(strategy='new')

    max_height = recent_heights[max_height_idx]
    min_height = recent_heights[min_height_idx]
    height_drop = max_height - min_height
    time_duration = (recent_timestamps[min_height_idx] - recent_timestamps[max_height_idx]) / 1000.0
    if height_drop < params['height_threshold'] or time_duration <= 0:
        return _empty_result(strategy='new')

    drop_velocity = height_drop / time_duration
    acceleration = 0.0
    if len(recent_points) >= 3:
        velocities = []
        for index in range(1, len(recent_points)):
            delta_t = (recent_timestamps[index] - recent_timestamps[index - 1]) / 1000.0
            if delta_t > 0:
                delta_z = recent_heights[index] - recent_heights[index - 1]
                velocities.append(delta_z / delta_t)
        if len(velocities) >= 2:
            delta_t = (recent_timestamps[-1] - recent_timestamps[-2]) / 1000.0
            if delta_t > 0:
                acceleration = abs((velocities[-1] - velocities[-2]) / delta_t)
    else:
        acceleration = abs(drop_velocity / time_duration)

    y_change = abs(recent_y_values[min_height_idx] - recent_y_values[max_height_idx])
    final_height = min_height
    low_height_duration = 0.0
    for index in range(min_height_idx, len(recent_points)):
        if recent_heights[index] < params['low_height_threshold']:
            if index < len(recent_points) - 1:
                low_height_duration += (recent_timestamps[index + 1] - recent_timestamps[index]) / 1000.0
            else:
                low_height_duration += (current_time_ms - recent_timestamps[index]) / 1000.0

    conditions = {
        'height_drop': height_drop >= params['height_threshold'],
        'velocity': drop_velocity > params['velocity_threshold'],
        'acceleration': acceleration > params['acceleration_threshold'],
        'final_height': final_height < params['low_height_threshold'],
        'height_window': params['max_height_min'] <= max_height <= params['max_height_max'],
        'max_height_positive': max_height > 0,
        'min_height_low': min_height < params['min_height_low'],
        'low_duration': low_height_duration > params['low_duration_threshold'],
    }

    if drop_velocity < params['slow_velocity_threshold']:
        conditions['velocity'] = False

    if len(recent_points) >= 3:
        last_heights = recent_heights[-3:]
        if len(last_heights) >= 2 and (last_heights[-1] - last_heights[0]) > 0.2:
            conditions['final_height'] = False

    satisfied_conditions = sum(conditions.values())
    if satisfied_conditions < params['min_conditions']:
        return _empty_result(strategy='new')

    return {
        'detected': True,
        'strategy': 'new',
        'message': f"检测到跌倒！(灵敏度: {sensitivity_level})",
        'metrics': {
            'sensitivity_level': sensitivity_level,
            'max_height': max_height,
            'min_height': min_height,
            'height_drop': height_drop,
            'drop_velocity': drop_velocity,
            'acceleration': acceleration,
            'y_change': y_change,
            'low_height_duration': low_height_duration,
            'satisfied_conditions': satisfied_conditions,
            'total_conditions': len(conditions),
            'min_conditions': params['min_conditions'],
        },
    }
