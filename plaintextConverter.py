import pandas as pd
import os
import struct
import numpy as np
from contextlib import ExitStack



packetStart1 = 0x90
packetStart2 = 0xEB  # packet headers
sysID = 0xC0  # any from 0-6
TmType = 0xC0
PMT_num = {0x09:1,0x0A:2,0x0B:3,0x0C:4}
baseFolder = "/home/wyatt/Projects/BOOMS/processed_data/"

def process_imager_packets(packet, size):
    seq_num = packet[:2]
    gondola_time = int.from_bytes(packet[2:8], byteorder="little")
    num = int(size / 7)
    # num of bytes containing events
    for i in range(num):
        event = packet[8 + i * 7:8 + i * 7 + 7]
        
        # header for this packet is b101011000, time is next 7 bits, then pmt counts at 10 bits each
        imager_time = int.from_bytes(event[0:2],byteorder="big") & 0b1111111
        # skip first 16 bits
        counts_packet = int.from_bytes(event[2:], byteorder="big")
        pmt4 = (counts_packet >> 0 ) & 0b1111111111
        pmt3 = (counts_packet >> 10) & 0b1111111111
        pmt2 = (counts_packet >> 20) & 0b1111111111
        pmt1 = (counts_packet >> 30) & 0b1111111111
        

        yield pmt1, pmt2, pmt3, pmt4, gondola_time,imager_time


def calc_mag_component(packet):
    # print(' '.join([f'{x:02x}' for x in packet]))
    
    Bx = 100 * (int.from_bytes(packet[2:5],
                               byteorder="big") - 0x800000) / 0x7FFFFF
    By = 100 * (int.from_bytes(packet[5:8],
                               byteorder="big") - 0x800000) / 0x7FFFFF
    Bz = 100 * (int.from_bytes(packet[8:11],
                               byteorder="big") - 0x800000) / 0x7FFFFF
    return Bx, By, Bz


def process_mag_field_packets(packet):
    gondola_time = int.from_bytes(packet[10:16], byteorder="little")
    quarter_second_readings = [
        packet[16:34],
        packet[34:52],
        packet[52:70],
        packet[70:70 + 18]]
    for i in range(4):
        Bx, By, Bz = calc_mag_component(quarter_second_readings[i])
        yield Bx, By, Bz, gondola_time


def process_gps_packet(packet):
    gondola_time = int.from_bytes(packet[10:16], byteorder="little")
    hour = int(packet[16])
    minute = int(packet[17])
    second = int(packet[18])
    fraction_second = int(packet[19])
    latitude = struct.unpack('d',packet[20:28])[0]
    longitude = struct.unpack('d',packet[28:36])[0]
    gps_quality = int(packet[36])
    num_sats = int(packet[37])
    precision_dilution = struct.unpack('f',packet[38:42])[0]
    altitude = int.from_bytes(packet[42:44], byteorder="big")
    geoid_sep = int.from_bytes(packet[44:46], byteorder="big")
    return gondola_time,hour,minute,second,fraction_second,latitude,longitude,gps_quality,num_sats,precision_dilution,altitude,geoid_sep

def process_imager_event_statistics(packet):
    num_events_list = [0]*8
    for i in range(8):
        pos = 16 + i * 6 
        event_in_imager = int.from_bytes(packet[pos : pos + 2],byteorder="little")
        num_events_list[i] = event_in_imager
    return num_events_list

def process_spectrometer_minor_frames(packet):
    #Accumulating into 100ms chuncks for now to reduce resultant file size
    detASpec = np.zeros(16)
    detBSpec = np.zeros(16)
    
    for i in range(5): 
        #spectra split up into a and b, 20ms chunks
        both = packet[6 + i * 40 : 6 + (i+1)*40]
        detASpec += np.array(Unpack(both[:20]))
        detBSpec += np.array(Unpack(both[20:]))
    return list(detASpec),list(detBSpec)
        
        
def Unpack(packed):
    chan00 =  (int(packed[ 0])         << 2) | (int(packed[ 1]) >> 6)
    chan01 = ((int(packed[ 1]) & 0x3F) << 4) | (int(packed[ 2]) >> 4)
    chan02 = ((int(packed[ 2]) & 0x0F) << 6) | (int(packed[ 3]) >> 2)
    chan03 = ((int(packed[ 3]) & 0x03) << 8) |  int(packed[ 4])
    chan04 =  (int(packed[ 5])         << 2) | (int(packed[ 6]) >> 6)
    chan05 = ((int(packed[ 6]) & 0x3F) << 4) | (int(packed[ 7]) >> 4)
    chan06 = ((int(packed[ 7]) & 0x0F) << 6) | (int(packed[ 8]) >> 2)
    chan07 = ((int(packed[ 8]) & 0x03) << 8) |  int(packed[ 9])
    chan08 =  (int(packed[10])         << 2) | (int(packed[11]) >> 6)
    chan09 = ((int(packed[11]) & 0x3F) << 4) | (int(packed[12]) >> 4)
    chan10 = ((int(packed[12]) & 0x0F) << 6) | (int(packed[13]) >> 2)
    chan11 = ((int(packed[13]) & 0x03) << 8) |  int(packed[14])
    chan12 =  (int(packed[15])         << 2) | (int(packed[16]) >> 6)
    chan13 = ((int(packed[16]) & 0x3F) << 4) | (int(packed[17]) >> 4)
    chan14 = ((int(packed[17]) & 0x0F) << 6) | (int(packed[18]) >> 2)
    chan15 = ((int(packed[18]) & 0x03) << 8) |  int(packed[19])
    return [chan00, chan01, chan02, chan03, chan04, chan05, chan06, chan07,
                    chan08, chan09, chan10, chan11, chan12, chan13, chan14, chan15]

def processGPSPPSPacket(packet):
    gondola_time = int.from_bytes(packet[10:16], byteorder="little")
    days = int(packet[16])
    hour = int(packet[17])
    minute = int(packet[18])
    second = int(packet[19])
    clockDiff = int.from_bytes(packet[20:24],byteorder="big")
    offset = int.from_bytes(packet[24:28],byteorder="big")
    
    return gondola_time,days,hour,minute,second,clockDiff,offset
        

def make_plaintext(rawFile,date_suffix=None):
    if date_suffix is None:
        date_suffix = rawFile[-24:-4]
    mag_file = f"{baseFolder}/magnetometer/mag_{date_suffix}.csv"
    imager_counts_filenames = [f"{baseFolder}/imager_data/imager{i}_{date_suffix}.csv" for i in range(7)]
    gps_file = f"{baseFolder}/gps/gps_{date_suffix}.csv"
    misc_file = f"{baseFolder}/misc/misc_{date_suffix}.csv"
    spectrometer_counts_file = f"{baseFolder}/spectrometer_data/spec_{date_suffix}.csv"
    imager_event_stats_file = f"{baseFolder}/imager_event_stats/imager_event_stats_{date_suffix}.csv"
    imager_housekeeping_file = f"{baseFolder}/imager_housekeeping/imager_housekeeping_{date_suffix}.csv"
    pps_file = f"{baseFolder}/pps/pps_{date_suffix}.csv"
    filenames = [mag_file,gps_file,misc_file,imager_event_stats_file,imager_housekeeping_file,pps_file,spectrometer_counts_file]
    
    for file in filenames:
        if os.path.exists(file):
            os.remove(file)
    for file in imager_counts_filenames:
        if os.path.exists(file):
            os.remove(file)

    with open(rawFile, "rb") as f:
        bytesRead = f.read()
    
    with ExitStack() as stack:
        print("Beginning file read")
        files = [stack.enter_context(open(filename,"a")) for filename in filenames]
        mag_f,gps_f,misc_f,imag_event_f,imag_hkpg_f ,pps_f ,spec_f= (files[i] for i in range(7))
        imagerEventFiles = [stack.enter_context(open(filename,"a")) for filename in imager_counts_filenames]
        
        for imagerFile in imagerEventFiles:
            imagerFile.write("PMT1,PMT2,PMT3,PMT4,gondola_time,imager_time,imager\n")
        pps_f.write("gondola_time,days,hour,minute,second,clockDiff,offset\n")
        spec_f.write("e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,gondola_time,spectrometer\n")
        
        num_bytes = len(bytesRead)
        imagerTimes = [0.0]*7
        secondsPassed = [0.0]*7
        previousTimes = [0.0]*7      
        i = 0
        while i < num_bytes:
            
            if bytesRead[i] == packetStart1 and bytesRead[i + 1] == packetStart2:
                # skip two bytes for checksum
                match list(bytesRead[i + 4:i + 6]): 
                    case [0xA0,0x02]:  # interface housekeeping packet
                        # print("interface hkpg")
                        pass

                    case [0xA0,0x0C]:  # interface imager event statistics packet
                        # print("event stats")
                        # checksumChecker(bytesRead[i:i+64])
                        event_info = process_imager_event_statistics(bytesRead[i:i+64])
                        imag_event_f.write(str(event_info)[1:-1])
                        imag_event_f.write("\n")
                        pass

                    case [0x60,0x60]:  # pps packet
                        pps_info = processGPSPPSPacket(bytesRead[i:i+28])
                        pps_f.write(str(pps_info)[1:-1])
                        pps_f.write("\n")                        

                    case [0x60,0x61]:  # gps position packet
                        gps_info = process_gps_packet(bytesRead[i:i + 46])
                        gps_f.write(str(gps_info)[1:-1])
                        gps_f.write("\n")

                    case [0x60,0x62]:  # gps velocity packet
                        speed = bytesRead[i + 24:i + 28]

                    case [0xB0,0xB0]:  
                        # magentometer packets
                        for mag_data in process_mag_field_packets(bytesRead[i:i + 88]):
                            mag_f.write(str(mag_data)[1:-1])
                            mag_f.write("\n")
                            
                    case [first,0xC0] if first in [0xC0,0xC1,0xC2,0xC3,0xC4,0xC5,0xC6]:
                        #imager packets
                        size = int.from_bytes(
                            bytesRead[i + 6:i + 8], byteorder="little")

                        # need to check the length
                        imager_packet = bytesRead[i + 8:i + 8 + size]
                        
                        imagerNumber = first & 0x0F

                        for imager_vals in process_imager_packets(imager_packet, size):
                            
                            imagerTimes[imagerNumber] = imager_vals[-1] / 125.0
                            
                            if imagerTimes[imagerNumber] - previousTimes[imagerNumber] < 0 :
                                #we have rolled over to the next second
                                secondsPassed[imagerNumber] += 1
                            previousTimes[imagerNumber] = imagerTimes[imagerNumber]
                            imagerTimes[imagerNumber] += secondsPassed[imagerNumber]
                            
                            imagerEventFiles[imagerNumber].write(str(imager_vals[0:-1])[1:-1])
                            imagerEventFiles[imagerNumber].write(f",{imagerTimes[imagerNumber]}")
                            imagerEventFiles[imagerNumber].write(f",{imagerNumber}")
                            imagerEventFiles[imagerNumber].write("\n")

                    case [first,last] if first in [0xC0,0xC1,0xC2,0xC3,0xC4,0xC5,0xC6] and last in [0x09,0x0A,0x0B,0x0C] :
                        size = int.from_bytes(bytesRead[i+6:i+8],byteorder="little") #11
                        packet = bytesRead[i+16:i+16+11]
                        low_level = int(packet[2])<<8 | int(packet[3])
                        peak_detect = int(packet[4])<<8 | int(packet[5])
                        high_level =  int(packet[6])<<8 | int(packet[7])
                        misc_f.write(str(first & 0x0F))
                        misc_f.write(",")
                        misc_f.write(str(PMT_num[last]))
                        misc_f.write(",")
                        misc_f.write(str(low_level))
                        misc_f.write(",")
                        misc_f.write(str(peak_detect))
                        misc_f.write(",")
                        misc_f.write(str(high_level))
                        misc_f.write("\n")

                    case [first,last] if first in [0xC0,0xC1,0xC2,0xC3,0xC4,0xC5,0xC6] and last in [0x0E] : 
                        # Imager Housekeeping
                        packet = bytesRead[i+16 + 2:i+16+18 + 2] #first 2 are header and seq num
                        funcs = [lambda x : x/10.0 -273.2, lambda x : x/10.0 -273.2,
                                 lambda x : x*.00132, lambda x: x* .00127,
                                 lambda x: x/10, lambda x: x/100,
                                 lambda x: x/10, lambda x : x*.00132]
                        imag_hkpg_f.write(str(first & 0x0F))
                        imag_hkpg_f.write(",")
                        for j in range(0,16,2):
                            val = int.from_bytes(packet[j:j+2],byteorder='big')
                            val = funcs[j//2](val)
                            imag_hkpg_f.write(str(val))
                            imag_hkpg_f.write(',')
                        imag_hkpg_f.write("\n")


                    case [first,0xD0] if first in  [0xD0,0xD1,0xD2,0xD3,0xD4,0xD5]:  
                        # spectrometer minor frames
                        size = int.from_bytes(bytesRead[i+6:i+8],byteorder="little")
                        box = first & 0x0F
                        specDict = {0:{'a':0,'b':1},1:{'a':2,'b':3},2:{'a':4,'b':5}}
                        num_frames = size // 212
                        gondola_time = int.from_bytes(bytesRead[i+10:i+16],byteorder="little") 
                        for frame in range(num_frames):
                            # print(bytesRead[(i + 16) + frame *212: (i + 16) + (frame+1) *212 ])
                            spec_a,spec_b = process_spectrometer_minor_frames(bytesRead[(i + 16) + frame *212: (i + 16) + (frame+1) *212 ])
                            spec_f.write(str(spec_a)[1:-1])
                            spec_f.write(",")
                            spec_f.write(str(gondola_time))
                            spec_f.write(",")
                            spec_f.write(str(specDict[box]['a']))
                            spec_f.write('\n')
                            spec_f.write(str(spec_b)[1:-1])
                            spec_f.write(",")
                            spec_f.write(str(gondola_time))
                            spec_f.write(",")
                            spec_f.write(str(specDict[box]['b']))
                            spec_f.write('\n')

                    case _:
                        pass
                i += int.from_bytes(bytesRead[i+6:i+8],byteorder="little") #increment by packet size
            else:
                i+=1


if __name__ == '__main__':
    # file = "/home/wyatt/Projects/BOOMS/flight_computer_data/tm_0872_20240505_131206.bin"
    # make_plaintext(file)
    base = "/home/wyatt/Projects/BOOMS/flight_computer_data/"
    allFiles = [base+file for file in os.listdir(base)]
    nums = [f"tm_0{i}" for i in range(976,980)]
    # nums = [f"tm_0872"]
    files = [file for file in allFiles if any(num in file for num in nums)]
    for file in files:
        make_plaintext(file)
    # gondola_time_conversion(file)
