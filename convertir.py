input_list = "/home/nazoku/Desktop/TFG/PEL4VAD/list/ucf/test.list"
output_list = "/home/nazoku/Desktop/TFG/PEL4VAD/list/ucf/corrected_test.list"

with open(input_list, 'r') as f_in, open(output_list, 'w') as f_out:
    for line in f_in:
        original = line.strip()
        if not original:
            f_out.write('\n')
            continue
        
        # Ejemplo: 'test/Abuse028_x264.npy' -> 'test/Abuse/Abuse028_x264.npy'
        if original.startswith('test/'):
            filename = original[5:]  # Quita 'test/' -> 'Abuse028_x264.npy'
            
            # Extrae la categoría del nombre del archivo
            if filename.startswith('Normal_Videos_'):
                category = 'Testing_Normal_Videos_Anomaly'
            else:
                import re
                match = re.match(r'^([A-Za-z]+)', filename)
                category = match.group(1) if match else 'Unknown'
            
            new_path = f"test/{category}/{filename}"
            f_out.write(new_path + '\n')
        else:
            f_out.write(original + '\n')

print(f"Lista corregida guardada como '{output_list}'")