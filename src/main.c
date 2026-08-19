/*
 * Copyright (c) 2021 Bosch Sensortec GmbH
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <stdio.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/atomic.h>
#include <math.h>
#define PI 3.14159265359f

// Constantes para filtro Madgwick 
#define deltat 0.01f                                             // Periodo de amostragem (1 ms = 0.001 s)
#define gyroMeasError 3.14159265358979f * (10.0f / 180.0f)         // erro de medição do giroscópio em rad/s (convertido de 5°/s)  
//#define beta sqrt(3.0f / 4.0f) * gyroMeasError                    // constante de correção do filtro Madgwick (proporcional ao erro de medição do giroscópio)  

float beta_val = 0.0f;
#define BETA_MOTION  (sqrt(3.0f / 4.0f) * (3.14159265358979f * (10.0f / 180.0f)))  // movimento normal
#define BETA_STATIC  0.5f   // parado: acelerômetro corrige forte

float SEq_1 = 1.0f, SEq_2 = 0.0f, SEq_3 = 0.0f, SEq_4 = 0.0f;     // elementos do quaternio de orientação estimada com condições iniciais

//Variáveis de offset
static float gyr_offset_x = 0.0f;
static float gyr_offset_y = 0.0f;
static float gyr_offset_z = 0.0f;

// Valor de thereshold para o gyro
#define GYRO_THRESHOLD 0.03f  // rad/s

// Acelerômetro filtrado (passa-baixa)
static float ax_f = 0.0f, ay_f = 0.0f, az_f = 1.0f;
#define ACC_ALPHA 0.1f  // 0.0 = muito suave, 1.0 = sem filtro — ajuste entre 0.05 e 0.2


/*Definido os botões de iniciação e parada*/
#define BUTTON_NODE DT_ALIAS(sw0)
static atomic_t app_running = ATOMIC_INIT(0);

K_SEM_DEFINE(run_sem, 0, 1);


#if !DT_NODE_HAS_STATUS(BUTTON_NODE, okay)
#error "Botão sw0 não encontrado"
#endif

static const struct gpio_dt_spec button =
    GPIO_DT_SPEC_GET(BUTTON_NODE, gpios);
#define BUTTON_NODE DT_ALIAS(sw0)

#if !DT_NODE_HAS_STATUS(BUTTON_NODE, okay)
#error "Botão sw0 não encontrado"
#endif

static struct gpio_callback button_cb;

void button_pressed(const struct device *dev,
                    struct gpio_callback *cb,
                    uint32_t pins)
{
     if (atomic_get(&app_running)) {
        /* Estava rodando → PAUSA */
        atomic_set(&app_running, 0);
    } else {
        /* Estava pausado → retorna */
        atomic_set(&app_running, 1);
    	printk("START\n");
    	k_sem_give(&run_sem);
    }
}

/* Função para botão de iniciar */
static void button_init(void)
{
    gpio_pin_configure_dt(&button, GPIO_INPUT);

    gpio_pin_interrupt_configure_dt(
        &button, GPIO_INT_EDGE_TO_ACTIVE);

    gpio_init_callback(&button_cb,
                       button_pressed,
                       BIT(button.pin));

    gpio_add_callback(button.port, &button_cb);
}

/* Filtro complementar para calculo do pitch e roll */
static float roll = 0.0f;
static float pitch = 0.0f;

#define DT     0.01f   /* 10 ms */
#define ALPHA  0.99f


/*-------Função filtro Madgwick-------*/
void filterUpdate(float w_x, float w_y, float w_z, float a_x, float a_y, float a_z)
{
// Variáveis para o filtro Madgwick

    float norm; // vetor norm
    float SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3, SEqDot_omega_4; // derivada do quaternio medida pelos giroscópios
    float f_1, f_2, f_3; // elementos da função objetivo
    float J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33; // elementos da jacobiana da função objetivo
    float SEqHatDot_1, SEqHatDot_2, SEqHatDot_3, SEqHatDot_4; // direção estimada do erro do giroscópio

// variaveis auxiliares para o evitar calculos repitidos
    float halfSEq_1 = 0.5f * SEq_1;
    float halfSEq_2 = 0.5f * SEq_2;
    float halfSEq_3 = 0.5f * SEq_3;
    float halfSEq_4 = 0.5f * SEq_4;
    float twoSEq_1 = 2.0f * SEq_1;
    float twoSEq_2 = 2.0f * SEq_2;
    float twoSEq_3 = 2.0f * SEq_3;

// Normalização das medidas do acelerometro
    norm = sqrt(a_x * a_x + a_y * a_y + a_z * a_z);
    a_x /= norm;
    a_y /= norm;
    a_z /= norm;

// Calculando os elementos da função objetivo e da jacobiana
    f_1 = twoSEq_2 * SEq_4- twoSEq_1 * SEq_3- a_x;
    f_2 = twoSEq_1 * SEq_2 + twoSEq_3 * SEq_4- a_y;
    f_3 = 1.0f- twoSEq_2 * SEq_2- twoSEq_3 * SEq_3- a_z;
    J_11or24 = twoSEq_3; // J_11 negado na multiplicação de matrizes
    J_12or23 = 2.0f * SEq_4; 
    J_13or22 = twoSEq_1; // J_12 negado na multiplicação de matrizes
    J_14or21 = twoSEq_2;
    J_32 = 2.0f * J_14or21; // negado na multiplicação de matrizes
    J_33 = 2.0f * J_11or24; // negado na multiplicação de matrizes

// Computando o gradiente (multiplicação de matriz)
    SEqHatDot_1 = J_14or21 * f_2- J_11or24 * f_1;
    SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2- J_32 * f_3;
    SEqHatDot_3 = J_12or23 * f_2- J_33 * f_3- J_13or22 * f_1;
    SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2;

// Normalização do gradiente
    norm = sqrt(SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4);
    SEqHatDot_1 /= norm;
    SEqHatDot_2 /= norm;
    SEqHatDot_3 /= norm;
    SEqHatDot_4 /= norm;

// Calculando a derrivada do quaternio medida pelo giroscópio
    SEqDot_omega_1 =-halfSEq_2 * w_x- halfSEq_3 * w_y- halfSEq_4 * w_z;
    SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z- halfSEq_4 * w_y;
    SEqDot_omega_3 = halfSEq_1 * w_y- halfSEq_2 * w_z + halfSEq_4 * w_x;
    SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y- halfSEq_3 * w_x;

// Calcula e depois integra a derrivada do quaternio estimada
    SEq_1 += (SEqDot_omega_1- (beta_val * SEqHatDot_1)) * deltat;
    SEq_2 += (SEqDot_omega_2- (beta_val * SEqHatDot_2)) * deltat;
    SEq_3 += (SEqDot_omega_3- (beta_val * SEqHatDot_3)) * deltat;
    SEq_4 += (SEqDot_omega_4- (beta_val * SEqHatDot_4)) * deltat;

// Normalização do quaternio
    norm = sqrt(SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4);
    SEq_1 /= norm;
    SEq_2 /= norm;
    SEq_3 /= norm;
    SEq_4 /= norm;
}
/* Função de calibração */ 
void calibrate_gyro(const struct device *dev)
{
    struct sensor_value gyr[3];
    const int N = 200;
    float sum_x = 0, sum_y = 0, sum_z = 0;

    printk("Calibrando giroscópio, mantenha o sensor parado...\n");

    for (int i = 0; i < N; i++) {
        sensor_sample_fetch(dev);
        sensor_channel_get(dev, SENSOR_CHAN_GYRO_XYZ, gyr);
        sum_x += sensor_value_to_double(&gyr[0]);
        sum_y += sensor_value_to_double(&gyr[1]);
        sum_z += sensor_value_to_double(&gyr[2]);
        k_sleep(K_MSEC(10));
    }

    gyr_offset_x = sum_x / N;
    gyr_offset_y = sum_y / N;
    gyr_offset_z = sum_z / N;

    printk("Offset: x=%.4f y=%.4f z=%.4f\n",
           gyr_offset_x, gyr_offset_y, gyr_offset_z);
}

/*Função de threshold*/
float deadband(float v) {
    return (fabsf(v) < GYRO_THRESHOLD) ? 0.0f : v;
}


int main(void)
{
    const struct device *dev = DEVICE_DT_GET_ONE(bosch_bmi270);
    struct sensor_value acc[3], gyr[3];
    struct sensor_value full_scale_accel, full_scale_gyro, sampling_freq, oversampling;


    printk("Aguardando inicio\n");

    button_init();

    if (!device_is_ready(dev)) {
        printk("BMI270 não pronto\n");
        return 0;
    }

    /*------ Configurando o sensor ------*/

    /*Acelerometro*/
    full_scale_accel.val1 = 2;  /*(g)*/
    full_scale_accel.val2 = 0;
    sampling_freq.val1 = 100;
    sampling_freq.val2 = 0;
    oversampling.val1 = 1;
    oversampling.val2 = 0;

    sensor_attr_set(dev, SENSOR_CHAN_ACCEL_XYZ,
                    SENSOR_ATTR_FULL_SCALE, &full_scale_accel);
    sensor_attr_set(dev, SENSOR_CHAN_ACCEL_XYZ,
                    SENSOR_ATTR_OVERSAMPLING, &oversampling);
    sensor_attr_set(dev, SENSOR_CHAN_ACCEL_XYZ,
                    SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);

    /*Gisroscópio*/                
    full_scale_gyro.val1 = 500;  /*(dps = degrees per second)*/

    sensor_attr_set(dev, SENSOR_CHAN_GYRO_XYZ,
                    SENSOR_ATTR_FULL_SCALE, &full_scale_gyro);
    sensor_attr_set(dev, SENSOR_CHAN_GYRO_XYZ,
                    SENSOR_ATTR_OVERSAMPLING, &oversampling);
    sensor_attr_set(dev, SENSOR_CHAN_GYRO_XYZ,
                    SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);

    calibrate_gyro(dev);

                    
    /* ------ LOOP PRINCIPAL ------ */
    while (1) {

        /* BLOQUEIA SE PAUSADO */
        if (!atomic_get(&app_running)) {
            printk("Aplicação pausada\n");
            k_sem_take(&run_sem, K_FOREVER);
            printk("Aplicação retomada\n");
        }

        /* LÊ SENSOR */
        sensor_sample_fetch(dev);
        sensor_channel_get(dev, SENSOR_CHAN_ACCEL_XYZ, acc);
        sensor_channel_get(dev, SENSOR_CHAN_GYRO_XYZ, gyr);

        /*printk("AX:%d.%04d AY:%d.%04d AZ:%d.%04d | "
               "GX:%d.%04d GY:%d.%04d GZ:%d.%04d\n\n",
               acc[0].val1, acc[0].val2,
               acc[1].val1, acc[1].val2,
               acc[2].val1, acc[2].val2,
               gyr[0].val1, gyr[0].val2,
               gyr[1].val1, gyr[1].val2,
               gyr[2].val1, gyr[2].val2); */

      

        /* Conversões para o filtro Madgwick */

        /*valores do acelerometro sem filtragem
        float ax = sensor_value_to_double(&acc[0]);
        float ay = sensor_value_to_double(&acc[1]);
        float az = sensor_value_to_double(&acc[2]);*/

        float ax_raw = sensor_value_to_double(&acc[0]);
        float ay_raw = sensor_value_to_double(&acc[1]);
        float az_raw = sensor_value_to_double(&acc[2]);

        // Passa-baixa exponencial
        ax_f = ACC_ALPHA * ax_raw + (1.0f - ACC_ALPHA) * ax_f;
        ay_f = ACC_ALPHA * ay_raw + (1.0f - ACC_ALPHA) * ay_f;
        az_f = ACC_ALPHA * az_raw + (1.0f - ACC_ALPHA) * az_f;

        float ax = ax_f;
        float ay = ay_f;
        float az = az_f;

        float gx = sensor_value_to_double(&gyr[0]) - gyr_offset_x; //rad/s
        float gy = sensor_value_to_double(&gyr[1]) - gyr_offset_y;         //Subtraindo o offset
        float gz = sensor_value_to_double(&gyr[2]) - gyr_offset_z;

        /* === ÂNGULOS PELO ACELERÔMETRO === */
        float roll_acc  = atan2f(ay, az) * 180.0f / PI;
        float pitch_acc = atan2f(-ax, sqrtf(ay*ay + az*az)) * 180.0f / PI;    

        /* === FILTRO COMPLEMENTAR === */
        roll  = ALPHA * (roll  + gx * DT) + (1.0f - ALPHA) * roll_acc;    /*Inclinação lateral*/
        pitch = ALPHA * (pitch + gy * DT) + (1.0f - ALPHA) * pitch_acc;   /*Inclinação frente/trás*/

        /*printk("Roll: %d.%02d deg | Pitch: %d.%02d deg\n\n",
            (int)roll, (int)(fabsf(roll) * 100) % 100,
            (int)pitch, (int)(fabsf(pitch) * 100) % 100); */

        /*Analisa os valores para verificar threshold*/    
        gx = deadband(gx);
        gy = deadband(gy);
        gz = deadband(gz);

        /* Atualiza o filtro Madgwick checando se o sensor está parado, se sim congela o quaternion*/
        bool is_static = (gx == 0.0f && gy == 0.0f && gz == 0.0f);

        // Adapta o beta conforme movimento
        beta_val = is_static ? BETA_STATIC : BETA_MOTION;

        if (!is_static) {
            filterUpdate(gx, gy, gz, ax, ay, az);
        }  

        printk("Q:%f,%f,%f,%f\n", SEq_1, SEq_2, SEq_3, SEq_4);
        printk("=====================================\n\n");

        //Teste
        printk("static=%d beta=%.3f gx=%.4f gy=%.4f gz=%.4f\n", is_static, beta_val, gx, gy, gz);


        k_sleep(K_MSEC(10));
    }
}
