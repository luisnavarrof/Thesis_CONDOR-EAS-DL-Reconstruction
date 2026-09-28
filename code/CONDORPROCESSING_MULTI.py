import panama as pn
import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CONDORDetectorGrid:
    """
    Clase para mapear coordenadas físicas a detectores específicos de CONDOR
    basado en la geometría real del observatorio
    """
    
    def __init__(self):
        # Dimensiones del detector individual
        self.detector_width = 8.1  # metros
        self.detector_height = 8.7  # metros
        
        # Configuración del área central: 10x10 = 100 detectores
        self.central_detectors_x = 10
        self.central_detectors_y = 10
        
        # Separación entre detectores centrales
        self.detector_spacing_x = 8.5  # metros (incluyendo gaps)
        self.detector_spacing_y = 9.0  # metros
        
        # Número de detectores periféricos
        self.peripheral_detectors = 20
        
        # Crear la grilla de detectores
        self.detector_positions = self._create_detector_grid()
        
    def _create_detector_grid(self):
        """Crear las posiciones de todos los detectores"""
        detectors = []
        detector_id = 0
        
        # === DETECTORES CENTRALES (10x10 = 100 detectores) ===
        # Centrar la grilla en (0,0)
        start_x = -(self.central_detectors_x * self.detector_spacing_x) / 2
        start_y = -(self.central_detectors_y * self.detector_spacing_y) / 2
        
        for i in range(self.central_detectors_x):
            for j in range(self.central_detectors_y):
                x_center = start_x + i * self.detector_spacing_x + self.detector_spacing_x/2
                y_center = start_y + j * self.detector_spacing_y + self.detector_spacing_y/2
                
                detectors.append({
                    'detector_id': detector_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'x_min': x_center - self.detector_width/2,
                    'x_max': x_center + self.detector_width/2,
                    'y_min': y_center - self.detector_height/2,
                    'y_max': y_center + self.detector_height/2,
                    'type': 'central'
                })
                detector_id += 1
        
        # === DETECTORES PERIFÉRICOS (20 detectores en líneas paralelas) ===
        # Los detectores periféricos forman líneas regulares paralelas al área central
        
        # Distancias específicas desde el borde del área central
        lateral_separation = 7.9  # metros desde los lados
        vertical_separation = 8.75  # metros desde arriba/abajo
        
        # Límites del área central
        central_half_width = (self.central_detectors_x * self.detector_spacing_x) / 2
        central_half_height = (self.central_detectors_y * self.detector_spacing_y) / 2
        
        # Posiciones de las líneas periféricas
        left_line_x = -central_half_width - lateral_separation
        right_line_x = central_half_width + lateral_separation
        top_line_y = central_half_height + vertical_separation
        bottom_line_y = -central_half_height - vertical_separation
        
        # Posiciones Y para detectores laterales (4 detectores cada lado)
        lateral_y_positions = np.linspace(-central_half_height + self.detector_spacing_y/2, 
                                         central_half_height - self.detector_spacing_y/2, 4)
        
        # Posiciones X para detectores superior/inferior (4 detectores cada lado)
        horizontal_x_positions = np.linspace(-central_half_width + self.detector_spacing_x/2, 
                                           central_half_width - self.detector_spacing_x/2, 4)
        
        peripheral_positions = []
        
        # Detectores superiores (100,101,102,103)
        for x in horizontal_x_positions:
            peripheral_positions.append((x, top_line_y))
        
        # Detectores inferiores (104,105,106,107)
        for x in horizontal_x_positions:
            peripheral_positions.append((x, bottom_line_y))
        
        # Detectores esquina inferior-izquierda (108)
        peripheral_positions.append((left_line_x, bottom_line_y))
        
        # Detectores esquina superior-izquierda (109)
        peripheral_positions.append((left_line_x, top_line_y))
        
        # Detectores laterales izquierdos (110,111,112,113)
        for y in lateral_y_positions:
            peripheral_positions.append((left_line_x, y))
        
        # Detectores laterales derechos (114,115,116,117)
        for y in lateral_y_positions:
            peripheral_positions.append((right_line_x, y))
        
        # Detectores esquina inferior-derecha (118)
        peripheral_positions.append((right_line_x, bottom_line_y))
        
        # Detectores esquina superior-derecha (119)
        peripheral_positions.append((right_line_x, top_line_y))
        
        # Crear detectores periféricos
        for i, (x_center, y_center) in enumerate(peripheral_positions):
            if i >= self.peripheral_detectors:
                break
                
            detectors.append({
                'detector_id': detector_id,
                'x_center': x_center,
                'y_center': y_center,
                'x_min': x_center - self.detector_width/2,
                'x_max': x_center + self.detector_width/2,
                'y_min': y_center - self.detector_height/2,
                'y_max': y_center + self.detector_height/2,
                'type': 'peripheral'
            })
            detector_id += 1
            
        return pd.DataFrame(detectors)
    
    def map_coordinates_to_detector(self, x, y):
        """
        Mapear coordenadas (x,y) al detector correspondiente
        Retorna detector_id o -1 si no está en ningún detector
        """
        # Vectorizar la búsqueda para eficiencia
        x_arr = np.array(x) if hasattr(x, '__iter__') else np.array([x])
        y_arr = np.array(y) if hasattr(y, '__iter__') else np.array([y])
        
        detector_ids = np.full(len(x_arr), -1, dtype=int)
        
        for _, detector in self.detector_positions.iterrows():
            # Encontrar partículas que caen en este detector
            mask = ((x_arr >= detector['x_min']) & (x_arr <= detector['x_max']) & 
                   (y_arr >= detector['y_min']) & (y_arr <= detector['y_max']))
            
            detector_ids[mask] = detector['detector_id']
        
        return detector_ids[0] if len(detector_ids) == 1 else detector_ids
    
    def create_binned_data(self, particles_df, time_bin_size=1.0, percentage_in_condor=None):
        """
        Crear datos binneados realísticamente por detector y tiempo
        """
        # Mapear coordenadas a detectores
        detector_ids = self.map_coordinates_to_detector(particles_df['x'], particles_df['y'])
        
        # Agregar detector_id al DataFrame
        particles_df = particles_df.copy()
        particles_df['detector_id'] = detector_ids
        
        # Filtrar partículas que no cayeron en ningún detector
        particles_in_detectors = particles_df[particles_df['detector_id'] != -1].copy()
        
        if len(particles_in_detectors) == 0:
            return pd.DataFrame(columns=['detector_id', 't_bin', 'particle_count', 'total_energy'])
        
        # Crear bins temporales
        particles_in_detectors['t_bin'] = np.floor(particles_in_detectors['t'] / time_bin_size).astype(int)
        
        # Agrupar por detector y bin temporal
        binned_data = particles_in_detectors.groupby(['detector_id', 't_bin']).agg({
            'E': ['count', 'sum', 'mean'],  # count = número de partículas, sum = energía total
            'x': 'mean',  # posición promedio en el detector
            'y': 'mean'
        }).reset_index()
        
        # Aplanar nombres de columnas
        binned_data.columns = ['detector_id', 't_bin', 'particle_count', 'total_energy', 'mean_energy', 'mean_x', 'mean_y']
        
        # Agregar información del detector
        binned_data = binned_data.merge(
            self.detector_positions[['detector_id', 'x_center', 'y_center', 'type']], 
            on='detector_id'
        )
        
        # Agregar columna percentage_in_condor
        if percentage_in_condor is not None:
            binned_data['percentage_in_condor'] = percentage_in_condor
        else:
            binned_data['percentage_in_condor'] = np.nan
        
        return binned_data

    
    def get_detector_info(self):
        """Obtener información resumen de los detectores"""
        central_count = len(self.detector_positions[self.detector_positions['type'] == 'central'])
        peripheral_count = len(self.detector_positions[self.detector_positions['type'] == 'peripheral'])
        
        info = {
            'total_detectores': len(self.detector_positions),
            'detectores_centrales': central_count,
            'detectores_perifericos': peripheral_count,
            'configuracion_central': f"{self.central_detectors_x}x{self.central_detectors_y}",
            'area_detector': self.detector_width * self.detector_height,
            'separacion_x': self.detector_spacing_x,
            'separacion_y': self.detector_spacing_y
        }
        
        return info

import argparse

parser = argparse.ArgumentParser(
    description="Convert CORSIKA DAT files (one primary energy) into CONDOR detector-binned CSVs."
)
parser.add_argument("--photon-dir", required=True, help="Folder with the photon DAT* files")
parser.add_argument("--proton-dir", required=True, help="Folder with the proton DAT* files")
parser.add_argument("--out-dir", required=True, help="Output root; photon/ and proton/ are created inside")
parser.add_argument("--energy", type=float, default=1e2, help="Primary energy of this batch in GeV (default 100)")
args = parser.parse_args()

energies = [args.energy]

base_directories = {
    "photon": args.photon_dir,
    "proton": args.proton_dir,
}

# Directorios de salida separados para cada tipo de partícula
output_directories = {
    "photon": os.path.join(args.out_dir, "photon"),
    "proton": os.path.join(args.out_dir, "proton"),
}


# Crear las carpetas base de salida si no existen
for particle_type, output_directory in output_directories.items():
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(os.path.join(output_directory, "realistic_binned_data"), exist_ok=True)

# Energías y ángulos
angles = list(range(0, 52, 2))  # 0° to 50°, step of 2

# Crear instancia del detector CONDOR
condor_detector = CONDORDetectorGrid()

# Mostrar información del detector
print("=== CONFIGURACIÓN DEL OBSERVATORIO CONDOR ===")
info = condor_detector.get_detector_info()
for key, value in info.items():
    print(f"{key.replace('_', ' ').title()}: {value}")

# Procesar archivos para cada tipo de partícula
for particle_type, input_directory in base_directories.items():
    print(f"\n=== PROCESANDO {particle_type.upper()} ===")
    dat_files = sorted(Path(input_directory).glob("DAT*"))
    particle_id = 1 if particle_type == "photon" else 14  # 1 para fotones, 14 para protones

    # Directorios de salida específicos
    output_directory = output_directories[particle_type]
    realistic_binned_directory = os.path.join(output_directory, "realistic_binned_data")

    # Mapear ángulos basado en grupos de 1000 archivos
    for i, dat_file in enumerate(dat_files):
        try:
            angle_index = i // 2000  # Determinar el grupo de ángulos
            angle = angles[angle_index]  # Obtener el ángulo correspondiente
            seed_number = i % 2000 + 1  # Número de semilla dentro del grupo

            # Leer archivo DAT
            run_header, event_header, particles = pn.read_DAT(glob=str(dat_file), mother_columns=False)
            print(f"Processing {dat_file} (Angle {angle}°, Seed {seed_number})")

            # Filtrar y procesar partículas
            particles['pdgid'] = particles['pdgid'].astype(int)
            particles_df = particles.query(
                "pdgid in (2212, -2212, 11, -11, 13, -13, 211, -211, 321, -321, 311, 3222, 3112, 3312, 3334, -3222, -3112, -3312, -3334)"
            )

            if len(particles_df) == 0:
                print(f"No particles found in {dat_file}")
                continue

            x = particles_df['x'] / 100
            y = particles_df['y'] / 100
            t = particles_df['t']
            En = particles_df['energy']

            # Normalizar el tiempo
            t -= t.min()

            # Preparar DataFrame de salida
            output_df = pd.DataFrame({
                'x': x,
                'y': y,
                't': t,
                'E': En
            })

            total_particles = len(output_df)

            # Filtrar por área de CONDOR (usando los límites del detector grid completo)
            detector_limits = condor_detector.detector_positions
            x_min_detector = detector_limits['x_min'].min()
            x_max_detector = detector_limits['x_max'].max()
            y_min_detector = detector_limits['y_min'].min()
            y_max_detector = detector_limits['y_max'].max()
            
            condor_df = output_df[
                (output_df['x'] >= x_min_detector) & (output_df['x'] <= x_max_detector) & 
                (output_df['y'] >= y_min_detector) & (output_df['y'] <= y_max_detector)
            ].reset_index(drop=True)
            
            if len(condor_df) == 0:
                print(f"No particles in detector area for {dat_file}")
                continue
            
            condor_df['t'] = condor_df['t'] - condor_df['t'].min()
            condor_particles = len(condor_df)
            condor_percentage = int((condor_particles / total_particles) * 100)

            # Calcular área de la shower
            x_min, x_max = condor_df['x'].min(), condor_df['x'].max()
            y_min, y_max = condor_df['y'].min(), condor_df['y'].max()
            area_shower = f"{int(x_max - x_min)} × {int(y_max - y_min)}"

            # Agregar metadatos
            condor_df['particle_id'] = particle_id
            condor_df['incidence_energy'] = energies[0]
            condor_df['incidence_angle'] = angle
            condor_df['area_shower_m2'] = area_shower
            condor_df['percentage_in_condor'] = condor_percentage
            condor_df['seed'] = seed_number
            condor_df['run'] = angle_index + 1

            new_column_order = ['x', 'y', 't', 'E', 'particle_id', 'incidence_energy', 'incidence_angle', 'area_shower_m2', 'percentage_in_condor', 'seed', 'run']
            condor_df = condor_df[new_column_order]

            # Ordenar por tiempo de llegada
            condor_df = condor_df.sort_values(by='t').reset_index(drop=True)

            # Crear subdirectorios para cada run
            run_realistic_binned_directory = os.path.join(realistic_binned_directory, f"run_{angle_index + 1}")
            os.makedirs(run_realistic_binned_directory, exist_ok=True)

            # ===== CREAR BINNING REALÍSTICO =====
            # Usar el detector CONDOR para crear bins realísticos
            time_bin_size = 1.0  # nanosegundos
            realistic_binned_data = condor_detector.create_binned_data(condor_df, time_bin_size)
            
            if len(realistic_binned_data) > 0:
                # Agregar metadatos a los datos binneados
                realistic_binned_data['particle_id'] = particle_id
                realistic_binned_data['incidence_energy'] = energies[0]
                realistic_binned_data['percentage_in_condor'] = condor_percentage
                realistic_binned_data['incidence_angle'] = angle
                realistic_binned_data['seed'] = seed_number
                realistic_binned_data['run'] = angle_index + 1
                
                # Reordenar columnas incluyendo percentage_in_condor
                realistic_column_order = [
                    'detector_id', 't_bin', 'particle_count', 'total_energy', 'mean_energy',
                    'mean_x', 'mean_y', 'x_center', 'y_center', 'type',
                    'percentage_in_condor', 'particle_id', 'incidence_energy', 'incidence_angle', 'seed', 'run'
                ]
                realistic_binned_data = realistic_binned_data[realistic_column_order]
                
                # Guardar datos binneados realísticamente
                realistic_binned_filename = f"realistic_binned_data_particleid_{particle_id}_energy_{energies[0]:.1E}_angle_{angle}_run_{angle_index + 1}_seed_{seed_number}.csv"
                realistic_binned_filepath = os.path.join(run_realistic_binned_directory, realistic_binned_filename)
                realistic_binned_data.to_csv(realistic_binned_filepath, index=False)
                
                # Estadísticas de detección
                central_detections = len(realistic_binned_data[realistic_binned_data['type'] == 'central'])
                peripheral_detections = len(realistic_binned_data[realistic_binned_data['type'] == 'peripheral'])
                unique_detectors = realistic_binned_data['detector_id'].nunique()
                
                print(f"  Particles in area: {condor_particles}")
                print(f"  Detectors hit: {unique_detectors}/120")
                print(f"  Central detector hits: {central_detections}")
                print(f"  Peripheral detector hits: {peripheral_detections}")
                print(f"  Total time bins: {realistic_binned_data['t_bin'].nunique()}")
                print(f"  Saved realistic binned data to {realistic_binned_filepath}")
            else:
                print(f"  No particles detected by any detector for {dat_file}")

        except Exception as e:
            print(f"Error processing {dat_file}: {e}")
            import traceback
            traceback.print_exc()

print("\n=== PROCESAMIENTO COMPLETADO ===")
print("Los datos han sido procesados con el sistema de binning realístico de CONDOR")
print("- Detectores centrales: IDs 0-99 (grilla 10x10)")
print("- Detectores periféricos: IDs 100-119 (20 detectores en configuración específica)")
print("- Binning temporal: 1 nanosegundo por bin")
print("- Archivos guardados en carpeta 'realistic_binned_data'")
