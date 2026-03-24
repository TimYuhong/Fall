import serial
import numpy as np
import time
import struct

DEBUG=False
MAGIC_WORD_ARRAY = np.array([2, 1, 4, 3, 6, 5, 8, 7])
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MSG_AZIMUT_STATIC_HEAT_MAP = 8
IQ_CHANNELS = 2


def _count_enabled_bits(mask):
    return bin(int(mask)).count("1")


class IWR6843AOP_TLV:
    def __init__(self, sdk_version=3.4,  cli_baud=115200,data_baud=921600, num_rx=4, num_tx=3,
                 verbose=False, connect=True, mode=0,cli_loc='COM9',data_loc='COM10',config_file=""):
        super(IWR6843AOP_TLV, self).__init__()
        self.connected = False
        self.verbose = verbose
        self.mode = mode
        if connect:
            # self.cli_port = serial.Serial(cli_loc, cli_baud)
            # self.data_port = serial.Serial(data_loc, data_baud)
            self.connected = True
        self.sdk_version = sdk_version
        self.num_rx_ant = num_rx
        self.num_tx_ant = num_tx
        self.num_virtual_ant = num_rx * num_tx
        self.config_file = config_file
        # if mode == 0:
        #     self._initialize(self.config_file)
    
    def _configure_radar(self, config):
        for i in config:
            self.cli_port.write((i + '\n').encode())
            # print(i)
            time.sleep(0.01)

    def _initialize(self, config_file):
        with open(config_file) as cfg_file:
            config = [line.rstrip('\r\n') for line in cfg_file]
        if self.connected:
            pass
            # self._configure_radar(config)

        self.config_params = {}  # Initialize an empty dictionary to store the configuration parameters
        chirp_tx_masks = []

        start_freq = None
        idle_time = None
        ramp_end_time = None
        freq_slope_const = None
        num_adc_samples = None
        num_adc_samples_round_to2 = None
        dig_out_sample_rate = None
        chirp_start_idx = None
        chirp_end_idx = None
        num_loops = None
        num_frames = None
        frame_periodicity = None

        num_rx_ant = self.num_rx_ant
        num_tx_ant = self.num_tx_ant
        aoa_fov = {
            "azimuth": (-90.0, 90.0),
            "elevation": (-90.0, 90.0),
        }

        for i in config:
            line = i.strip()
            if not line or line.startswith("%"):
                continue

            # Split the line
            split_words = line.split()

            # Get the information about the profile configuration
            if "profileCfg" in split_words[0]:
                start_freq = float(split_words[2])
                idle_time = float(split_words[3])
                ramp_end_time = float(split_words[5])
                freq_slope_const = float(split_words[8])
                num_adc_samples = int(split_words[10])
                num_adc_samples_round_to2 = 1

                while num_adc_samples > num_adc_samples_round_to2:
                    num_adc_samples_round_to2 = num_adc_samples_round_to2 * 2

                dig_out_sample_rate = int(split_words[11])

            elif "channelCfg" in split_words[0]:
                num_rx_ant = max(_count_enabled_bits(split_words[1]), 1)
                num_tx_ant = max(_count_enabled_bits(split_words[2]), 1)

            elif "chirpCfg" in split_words[0]:
                chirp_tx_masks.append(
                    (
                        int(split_words[1]),
                        int(split_words[2]),
                        int(split_words[8]),
                    )
                )

            # Get the information about the frame configuration    
            elif "frameCfg" in split_words[0]:

                chirp_start_idx = int(split_words[1])
                chirp_end_idx = int(split_words[2])
                num_loops = int(split_words[3])
                num_frames = int(split_words[4])
                frame_periodicity = float(split_words[5])

            elif "aoaFovCfg" in split_words[0]:
                aoa_fov = {
                    "azimuth": (float(split_words[2]), float(split_words[3])),
                    "elevation": (float(split_words[4]), float(split_words[5])),
                }

        if None in (
            start_freq,
            idle_time,
            ramp_end_time,
            freq_slope_const,
            num_adc_samples,
            num_adc_samples_round_to2,
            dig_out_sample_rate,
            chirp_start_idx,
            chirp_end_idx,
            num_loops,
        ):
            raise ValueError(f"cfg parse failed for {config_file}")

        tx_mask_in_frame = 0
        for cfg_start, cfg_end, tx_mask in chirp_tx_masks:
            if cfg_end < chirp_start_idx or cfg_start > chirp_end_idx:
                continue
            tx_mask_in_frame |= tx_mask
        if tx_mask_in_frame:
            num_tx_ant = _count_enabled_bits(tx_mask_in_frame)

        # Combine the read data to obtain the configuration parameters
        num_chirps_per_frame = (chirp_end_idx - chirp_start_idx + 1) * num_loops
        num_doppler_bins = int(num_chirps_per_frame / num_tx_ant)
        range_resolution_m = (3e8 * dig_out_sample_rate * 1e3) / (
            2 * freq_slope_const * 1e12 * num_adc_samples
        )
        range_idx_to_m = (3e8 * dig_out_sample_rate * 1e3) / (
            2 * freq_slope_const * 1e12 * num_adc_samples_round_to2
        )
        doppler_resolution_mps = 3e8 / (
            2
            * start_freq
            * 1e9
            * (idle_time + ramp_end_time)
            * 1e-6
            * num_doppler_bins
            * num_tx_ant
        )
        max_range = (300 * 0.9 * dig_out_sample_rate) / (2 * freq_slope_const * 1e3)
        max_velocity = 3e8 / (
            4 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * num_tx_ant
        )
        frame_length = int(
            num_adc_samples
            * num_chirps_per_frame
            * num_rx_ant
            * IQ_CHANNELS
        )

        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant
        self.num_virtual_ant = num_rx_ant * num_tx_ant

        self.config_params["config_file"] = config_file
        self.config_params["num_tx"] = num_tx_ant
        self.config_params["num_rx"] = num_rx_ant
        self.config_params["num_loops"] = num_loops
        self.config_params["num_adc_samples_raw"] = num_adc_samples
        self.config_params["num_chirps_per_frame"] = num_chirps_per_frame
        self.config_params["num_doppler_bins"] = num_doppler_bins
        self.config_params["num_range_bins"] = num_adc_samples_round_to2
        self.config_params["range_resolution_m"] = range_resolution_m
        self.config_params["doppler_resolution_mps"] = doppler_resolution_mps
        self.config_params["frame_length"] = frame_length
        self.config_params["aoa_fov"] = aoa_fov
        self.config_params["frame_periodicity_ms"] = frame_periodicity
        self.config_params["num_frames"] = num_frames
        self.config_params["chirp_start_idx"] = chirp_start_idx
        self.config_params["chirp_end_idx"] = chirp_end_idx
        self.config_params["numDopplerBins"] = num_doppler_bins
        self.config_params["numRangeBins"] = num_adc_samples_round_to2
        self.config_params["rangeResolutionMeters"] = round(range_resolution_m, 2)
        self.config_params["rangeIdxToMeters"] = range_idx_to_m
        self.config_params["dopplerResolutionMps"] = round(doppler_resolution_mps, 2)
        self.config_params["maxRange"] = round(max_range, 2)
        self.config_params["maxVelocity"] = round(max_velocity, 2)
        return self.config_params


    def close(self):
        """End connection between radar and machine

        Returns:
            None

        """
        self.cli_port.write('sensorStop\n'.encode())
        self.cli_port.close()
        self.data_port.close()

    def _read_buffer(self):
        """

        Returns:

        """
        byte_buffer = self.data_port.read(self.data_port.in_waiting)

        return byte_buffer

    def _parse_header_data(self, byte_buffer, idx):
        """Parses the byte buffer for the header of the data

        Args:
            byte_buffer: Buffer with TLV data
            idx: Current reading index of the byte buffer

        Returns:
            Tuple [Tuple (int), int]

        """
        magic, idx = self._unpack(byte_buffer, idx, order='>', items=1, form='Q')
        (version, length, platform, frame_num, cpu_cycles, num_obj, num_tlvs), idx = self._unpack(byte_buffer, idx,
                                                                                                    items=7, form='I')
        subframe_num, idx = self._unpack(byte_buffer, idx, items=1, form='I')
        return (version, length, platform, frame_num, cpu_cycles, num_obj, num_tlvs, subframe_num), idx
    
    def _parse_header_tlv(self, byte_buffer, idx):
        """ Parses the byte buffer for the header of a tlv

        """
        (tlv_type, tlv_length), idx = self._unpack(byte_buffer, idx, items=2, form='I')
        return (tlv_type, tlv_length), idx

    def _parse_msg_detected_points(self, byte_buffer, idx):
        """ Parses the information of the detected points message

        """
        (x,y,z,vel), idx = self._unpack(byte_buffer, idx, items=4, form='f')
       
        return (x,y,z,vel), idx

    def _parse_msg_detected_points_side_info(self,byte_buffer, idx):
        (snr,noise), idx = self._unpack(byte_buffer, idx, items=2, form='H')
        return (snr,noise),idx

    def _parse_msg_azimut_static_heat_map(self, byte_buffer, idx):
        """ Parses the information of the azimuth heat map

        """
        try:
            (imag, real), idx = self._unpack(byte_buffer, idx, items=2, form='H')
            return (imag, real), idx
        except TypeError:
            pass

   
    def _process_azimut_heat_map(self, byte_buffer):
            """
            热图
            """
            idx = byte_buffer.index(MAGIC_WORD)
            header_data, idx = self._parse_header_data(byte_buffer, idx)    
            # print(header_data,idx)
            (tlv_type, tlv_length), idx = self._parse_header_tlv(byte_buffer, idx)
            # print(tlv_type, tlv_length,idx)
            azimuth_map = np.zeros((self.num_virtual_ant, self.config_params['numRangeBins'], 2),dtype=np.int16)
            # azimuth_map = np.zeros((7, self.config_params['numRangeBins'], 2),dtype=np.int16)
            for bin_idx in range(self.config_params['numRangeBins']):
                # for ant in range(7):
                for ant in range(self.num_virtual_ant):
                    azimuth_map[ant][bin_idx][:], idx = self._parse_msg_azimut_static_heat_map(byte_buffer, idx)
            return azimuth_map

    def _process_detected_points(self, byte_buffer):
            """
            点云
            """
            idx = byte_buffer.index(MAGIC_WORD)
            header_data, idx = self._parse_header_data(byte_buffer, idx)    
            # print(header_data,idx)
  
            num_tlvs=header_data[6]
            
            ####  tvl1  ####
            (tlv_type, tlv_length), idx = self._parse_header_tlv(byte_buffer, idx)
            num_points=int(tlv_length/16)
            data=np.zeros((num_points,6),dtype=np.float)
            for i in range(num_points):
                ( x, y, z,vel), idx = self._parse_msg_detected_points(byte_buffer, idx)
                data[i][0]=x
                data[i][1]=y
                data[i][2]=z
                data[i][3]=vel
            
            (tlv_type, tlv_length), idx = self._parse_header_tlv(byte_buffer, idx)
            for i in range(num_points):
                (snr,noise), idx = self._parse_msg_detected_points_side_info(byte_buffer, idx)
                data[i][4]=snr
                data[i][5]=noise

            return data
    @staticmethod
    def _unpack(byte_buffer, idx, order='', items=1, form='I'):
        """Helper function for parsing binary byte data

        Args:
            byte_buffer: Buffer with data
            idx: Curex in the buffer
            order: Little endian or big endian
            items: Number of items to be extracted
            form: Data type to be extracted

        Returns:rent ind
            Tuple [Tuple (object), int]

        """
        size = {'H': 2, 'h': 2, 'I': 4, 'Q': 8, 'f': 4}
        try:
            data = struct.unpack(order + str(items) + form, byte_buffer[idx:idx + (items * size[form])])
            if len(data) == 1:
                data = data[0]
            return data, idx + (items * size[form])
        except:
            return None
