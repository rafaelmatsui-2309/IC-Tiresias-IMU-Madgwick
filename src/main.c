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
#define deltat 0.001f                                             // sampling period in seconds (shown as 1 ms)
#define gyroMeasError 3.14159265358979f * (5.0f / 180.0f)         // gyroscope measurement error in rad/s (shown as 5 deg/s)  
#define beta sqrt(3.0f / 4.0f) * gyroMeasError                    // compute beta  

float SEq_1 = 1.0f, SEq_2 = 0.0f, SEq_3 = 0.0f, SEq_4 = 0.0f;     // estimated orientation quaternion elements with initial conditions


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
// Local system variables

    float norm; // vector norm
    float SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3, SEqDot_omega_4; // quaternion derrivative from gyroscopes elements
    float f_1, f_2, f_3; // objective function elements
    float J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33; // objective function Jacobian elements
    float SEqHatDot_1, SEqHatDot_2, SEqHatDot_3, SEqHatDot_4; // estimated direction of the gyroscope error

// Axulirary variables to avoid reapeated calcualtions
    float halfSEq_1 = 0.5f * SEq_1;
    float halfSEq_2 = 0.5f * SEq_2;
    float halfSEq_3 = 0.5f * SEq_3;
    float halfSEq_4 = 0.5f * SEq_4;
    float twoSEq_1 = 2.0f * SEq_1;
    float twoSEq_2 = 2.0f * SEq_2;
    float twoSEq_3 = 2.0f * SEq_3;

// Normalise the accelerometer measurement
    norm = sqrt(a_x * a_x + a_y * a_y + a_z * a_z);
    a_x /= norm;
    a_y /= norm;
    a_z /= norm;

// Compute the objective function and Jacobian
    f_1 = twoSEq_2 * SEq_4- twoSEq_1 * SEq_3- a_x;
    f_2 = twoSEq_1 * SEq_2 + twoSEq_3 * SEq_4- a_y;
    f_3 = 1.0f- twoSEq_2 * SEq_2- twoSEq_3 * SEq_3- a_z;
    J_11or24 = twoSEq_3; // J_11 negated in matrix multiplication
    J_12or23 = 2.0f * SEq_4; 
    J_13or22 = twoSEq_1; // J_12 negated in matrix multiplication
    J_14or21 = twoSEq_2;
    J_32 = 2.0f * J_14or21; // negated in matrix multiplication
    J_33 = 2.0f * J_11or24; // negated in matrix multiplication

// Compute the gradient (matrix multiplication)
    SEqHatDot_1 = J_14or21 * f_2- J_11or24 * f_1;
    SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2- J_32 * f_3;
    SEqHatDot_3 = J_12or23 * f_2- J_33 * f_3- J_13or22 * f_1;
    SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2;

// Normalise the gradient
    norm = sqrt(SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4);
    SEqHatDot_1 /= norm;
    SEqHatDot_2 /= norm;
    SEqHatDot_3 /= norm;
    SEqHatDot_4 /= norm;

// Compute the quaternion derrivative measured by gyroscopes
    SEqDot_omega_1 =-halfSEq_2 * w_x- halfSEq_3 * w_y- halfSEq_4 * w_z;
    SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z- halfSEq_4 * w_y;
    SEqDot_omega_3 = halfSEq_1 * w_y- halfSEq_2 * w_z + halfSEq_4 * w_x;
    SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y- halfSEq_3 * w_x;

// Compute then integrate the estimated quaternion derrivative
    SEq_1 += (SEqDot_omega_1- (beta * SEqHatDot_1)) * deltat;
    SEq_2 += (SEqDot_omega_2- (beta * SEqHatDot_2)) * deltat;
    SEq_3 += (SEqDot_omega_3- (beta * SEqHatDot_3)) * deltat;
    SEq_4 += (SEqDot_omega_4- (beta * SEqHatDot_4)) * deltat;

// Normalise quaternion
    norm = sqrt(SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4);
    SEq_1 /= norm;
    SEq_2 /= norm;
    SEq_3 /= norm;
    SEq_4 /= norm;
}

int main(void)
{
    const struct device *dev = DEVICE_DT_GET_ONE(bosch_bmi270);
    struct sensor_value acc[3], gyr[3];
    struct sensor_value full_scale_accel, full_scale_gyro, sampling_freq, oversampling;
    //float SEq_1 = 1.0f, SEq_2 = 0.0f, SEq_3 = 0.0f, SEq_4 = 0.0f;  // estimated orientation quaternion elements with initial conditions


    printk("Aguardando inicio\n");

    button_init();

    if (!device_is_ready(dev)) {
        printk("BMI270 não pronto\n");
        return 0;
    }

    /* Configurando o sensor */
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

    /* === LOOP PRINCIPAL === */
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

        /* === ACELERÔMETRO (m/s² → g) === 
        float ax = sensor_value_to_double(&acc[0]) / 9.80665;
        float ay = sensor_value_to_double(&acc[1]) / 9.80665;
        float az = sensor_value_to_double(&acc[2]) / 9.80665;

           === GIROSCÓPIO (rad/s → deg/s) === 
        float gx = sensor_value_to_double(&gyr[0]) * 180.0f / PI;
        float gy = sensor_value_to_double(&gyr[1]) * 180.0f / PI; */

        /* Conversões para o filtro Madgwick */
        float ax = sensor_value_to_double(&acc[0]);
        float ay = sensor_value_to_double(&acc[1]);
        float az = sensor_value_to_double(&acc[2]);

        float gx = sensor_value_to_double(&gyr[0]); // rad/s
        float gy = sensor_value_to_double(&gyr[1]);
        float gz = sensor_value_to_double(&gyr[2]);

        /* === ÂNGULOS PELO ACELERÔMETRO === */
        float roll_acc  = atan2f(ay, az) * 180.0f / PI;
        float pitch_acc = atan2f(-ax, sqrtf(ay*ay + az*az)) * 180.0f / PI;    

        /* === FILTRO COMPLEMENTAR === */
        roll  = ALPHA * (roll  + gx * DT) + (1.0f - ALPHA) * roll_acc;    /*Inclinação lateral*/
        pitch = ALPHA * (pitch + gy * DT) + (1.0f - ALPHA) * pitch_acc;   /*Inclinação frente/trás*/

        /*printk("Roll: %d.%02d deg | Pitch: %d.%02d deg\n\n",
            (int)roll, (int)(fabsf(roll) * 100) % 100,
            (int)pitch, (int)(fabsf(pitch) * 100) % 100); */   

        /* Atualiza o filtro Madgwick */
        filterUpdate(gx, gy, gz, ax, ay, az);    

        printk("Q:%f,%f,%f,%f\n", SEq_1, SEq_2, SEq_3, SEq_4);


        k_sleep(K_MSEC(1));
    }
}
