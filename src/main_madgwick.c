/*
 * IMU Visualizer — Filtro Madgwick Combinado
 * Hardware: nRF5340 Audio DK + BMI270
 *
 * Este firmware combina o melhor dos dois filtros:
 *
 * DO PROJETO original:
 *   ✓ Versão otimizada do Madgwick (sem Jacobiana explícita)
 *   ✓ Estado interno em quaternion puro (sem passar por Euler)
 *   ✓ Guarda contra acelerômetro zero (evita NaN)
 *   ✓ dt real via k_uptime_get()
 *
 * DO SEU PROJETO:
 *   ✓ Calibração de offset do giroscópio
 *   ✓ Filtro passa-baixa no acelerômetro
 *   ✓ Beta adaptativo — baixo no movimento, alto quando parado
 *   ✓ Botão sw0 para pausar/retomar
 *
 * RESULTADO ESPERADO:
 *   - Sem drift quando parado (beta alto corrige via acelerômetro)
 *   - Sem congelamento de erro (nunca trava o filtro)
 *   - Sem gimbal lock (quaternion puro, sem Euler interno)
 *   - Resposta suave no movimento (versão otimizada do original)
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/atomic.h>
#include <stdio.h>
#include <math.h>

/* ===========================================================
 * PARÂMETROS DO FILTRO
 * ===========================================================
 *
 * Beta adaptativo — a principal contribuição do seu projeto:
 *
 * BETA_MOTION → usado durante movimento detectado
 *   Valor baixo: giroscópio domina, resposta rápida e suave
 *   Recomendação: 0.03 a 0.07
 *
 * BETA_STATIC → usado quando parado
 *   Valor alto: acelerômetro domina, corrige drift acumulado
 *   Recomendação: 0.3 a 0.5
 *
 * A diferença entre os dois é a chave:
 *   - Código original usava 0.03 fixo → drift parado
 *   - Seu código congelava o filtro → não voltava à posição
 *   - Aqui: beta alto quando parado corrige SEM congelar
 *
 * GYRO_THRESHOLD → detecta se o sensor está parado
 *   Só muda o beta, NÃO zera o giroscópio nem congela o filtro
 *   O filtro sempre roda — a diferença é só o peso do acelerômetro
 *
 * ACC_ALPHA → suavização do acelerômetro 
 *   Reduz ruído de alta frequência sem atrasar muito a resposta
 */
#define BETA_MOTION    0.04f   /* movimento: giroscópio domina        */
#define BETA_STATIC    0.4f    /* parado: acelerômetro corrige drift   */
#define GYRO_THRESHOLD 0.03f   /* rad/s — limiar de detecção de repouso */
#define ACC_ALPHA      0.1f    /* passa-baixa do acelerômetro          */

/* Quaternion de estado — nunca convertido para Euler internamente */
static float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;

/* Acelerômetro filtrado (passa-baixa) */
static float ax_f = 0.0f, ay_f = 0.0f, az_f = 1.0f;

/* ===========================================================
 * FILTRO MADGWICK COMBINADO
 * ===========================================================
 *
 * Implementação otimizada com beta adaptativo do seu.
 *
 * Parâmetro beta recebido externamente — permite adaptar
 * o comportamento sem modificar a função do filtro.
 *
 * Etapas:
 *   1. Derivada do quaternion pelo giroscópio (qDot)
 *   2. Guarda acelerômetro zero — evita NaN
 *   3. Normaliza acelerômetro
 *   4. Gradiente otimizado — sem Jacobiana explícita
 *   5. Aplica beta × gradiente em qDot
 *   6. Integra com dt real
 *   7. Normaliza quaternion
 */
static void madgwick_update(float gx, float gy, float gz,
                             float ax, float ay, float az,
                             float dt, float beta)
{
    float recipNorm;
    float s0, s1, s2, s3;
    float qDot1, qDot2, qDot3, qDot4;

    /* ── Etapa 1: derivada pelo giroscópio ─────────────────
     * q_dot = 0.5 * q ⊗ [0, gx, gy, gz]
     */
    qDot1 = 0.5f * (-q1*gx - q2*gy - q3*gz);
    qDot2 = 0.5f * ( q0*gx + q2*gz - q3*gy);
    qDot3 = 0.5f * ( q0*gy - q1*gz + q3*gx);
    qDot4 = 0.5f * ( q0*gz + q1*gy - q2*gx);

    /* ── Etapa 2-5: correção pelo acelerômetro ─────────────
     * Só executa se acelerômetro for válido
     */
    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {

        /* Normaliza acelerômetro */
        recipNorm = 1.0f / sqrtf(ax*ax + ay*ay + az*az);
        ax *= recipNorm;
        ay *= recipNorm;
        az *= recipNorm;

        /* Variáveis auxiliares */
        float _2q0 = 2.0f * q0;
        float _2q1 = 2.0f * q1;
        float _2q2 = 2.0f * q2;
        float _2q3 = 2.0f * q3;
        float _4q0 = 4.0f * q0;
        float _4q1 = 4.0f * q1;
        float _4q2 = 4.0f * q2;
        float _8q1 = 8.0f * q1;
        float _8q2 = 8.0f * q2;
        float q0q0 = q0 * q0;
        float q1q1 = q1 * q1;
        float q2q2 = q2 * q2;
        float q3q3 = q3 * q3;

        /* Gradiente descendente otimizado 
         *
         * Matematicamente equivalente a J^T * f do seu código,
         * mas sem instanciar J e f separadamente.
         * s0..s3 = erro entre orientação estimada e gravidade medida.
         */
        s0 = _4q0*q2q2 + _2q2*ax + _4q0*q1q1 - _2q1*ay;

        s1 = _4q1*q3q3 - _2q3*ax + 4.0f*q0q0*q1 - _2q0*ay
           - _4q1 + _8q1*q1q1 + _8q1*q2q2 + _4q1*az;

        s2 = 4.0f*q0q0*q2 + _2q0*ax + _4q2*q3q3 - _2q3*ay
           - _4q2 + _8q2*q1q1 + _8q2*q2q2 + _4q2*az;

        s3 = 4.0f*q1q1*q3 - _2q1*ax + 4.0f*q2q2*q3 - _2q2*ay;

        /* Normaliza o gradiente */
        recipNorm = 1.0f / sqrtf(s0*s0 + s1*s1 + s2*s2 + s3*s3);
        s0 *= recipNorm;
        s1 *= recipNorm;
        s2 *= recipNorm;
        s3 *= recipNorm;

        /* Aplica feedback com beta adaptativo 
         *
         * Beta baixo (MOTION): acelerômetro corrige pouco
         *   → giroscópio domina → resposta rápida e suave
         *
         * Beta alto (STATIC): acelerômetro corrige muito
         *   → puxa quaternion de volta à gravidade real
         *   → elimina drift acumulado sem congelar o filtro
         */
        qDot1 -= beta * s0;
        qDot2 -= beta * s1;
        qDot3 -= beta * s2;
        qDot4 -= beta * s3;
    }

    /* ── Etapa 6: integração com dt real ────────────
     * Mais preciso que DT fixo — compensa variações do loop
     */
    q0 += qDot1 * dt;
    q1 += qDot2 * dt;
    q2 += qDot3 * dt;
    q3 += qDot4 * dt;

    /* ── Etapa 7: normaliza quaternion ─────────────────────*/
    recipNorm = 1.0f / sqrtf(q0*q0 + q1*q1 + q2*q2 + q3*q3);
    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;
}

/* ===========================================================
 * DETECÇÃO DE REPOUSO
 * ===========================================================
 *
 * Verifica se o sensor está parado baseado na magnitude
 * total do giroscópio — mais robusto que comparar cada
 * eixo individualmente com zero.
 *
 * Retorna true se o sensor estiver em repouso.
 * NÃO zera o giroscópio — o filtro sempre recebe o valor real.
 * Só muda o beta para aumentar a influência do acelerômetro.
 */
static bool is_sensor_static(float gx, float gy, float gz)
{
    float mag = sqrtf(gx*gx + gy*gy + gz*gz);
    return mag < GYRO_THRESHOLD;
}

/* ===========================================================
 * BOTÃO sw0 — pausar / retomar 
 * ===========================================================
 */
#define BUTTON_NODE DT_ALIAS(sw0)

#if !DT_NODE_HAS_STATUS(BUTTON_NODE, okay)
#error "Botão sw0 não encontrado no Device Tree"
#endif

static const struct gpio_dt_spec button =
    GPIO_DT_SPEC_GET(BUTTON_NODE, gpios);
static struct gpio_callback button_cb;
static atomic_t app_running = ATOMIC_INIT(0);
K_SEM_DEFINE(run_sem, 0, 1);

static void button_pressed(const struct device *dev,
                            struct gpio_callback *cb,
                            uint32_t pins)
{
    if (atomic_get(&app_running)) {
        atomic_set(&app_running, 0);
    } else {
        atomic_set(&app_running, 1);
        printk("START\n");
        k_sem_give(&run_sem);
    }
}

static void button_init(void)
{
    gpio_pin_configure_dt(&button, GPIO_INPUT);
    gpio_pin_interrupt_configure_dt(&button, GPIO_INT_EDGE_TO_ACTIVE);
    gpio_init_callback(&button_cb, button_pressed, BIT(button.pin));
    gpio_add_callback(button.port, &button_cb);
}

/* ===========================================================
 * CALIBRAÇÃO DO GIROSCÓPIO 
 * ===========================================================
 *
 * Mede o bias em repouso e subtrai de todas as leituras.
 * Com beta adaptativo isso é ainda mais importante — um bias
 * não calibrado vai causar drift mesmo com BETA_STATIC alto,
 * pois o acelerômetro só corrige roll e pitch, não yaw.
 */
static float gyr_offset_x = 0.0f;
static float gyr_offset_y = 0.0f;
static float gyr_offset_z = 0.0f;

static void calibrate_gyro(const struct device *dev)
{
    struct sensor_value gyr[3];
    const int N = 200;
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

    printk("Calibrando giroscópio — mantenha o sensor parado...\n");

    for (int i = 0; i < N; i++) {
        sensor_sample_fetch(dev);
        sensor_channel_get(dev, SENSOR_CHAN_GYRO_XYZ, gyr);
        sum_x += sensor_value_to_double(&gyr[0]);
        sum_y += sensor_value_to_double(&gyr[1]);
        sum_z += sensor_value_to_double(&gyr[2]);
        k_sleep(K_MSEC(10));
    }

    gyr_offset_x = (float)(sum_x / N);
    gyr_offset_y = (float)(sum_y / N);
    gyr_offset_z = (float)(sum_z / N);

    printk("Offset: x=%.4f y=%.4f z=%.4f\n",
           (double)gyr_offset_x,
           (double)gyr_offset_y,
           (double)gyr_offset_z);
}

/* ===========================================================
 * MAIN
 * ===========================================================
 */
int main(void)
{
    const struct device *dev = DEVICE_DT_GET_ONE(bosch_bmi270);
    struct sensor_value acc[3], gyr[3];
    struct sensor_value full_scale_accel, full_scale_gyro,
                        sampling_freq, oversampling;

    printk("IMU Madgwick Combinado iniciando...\n");

    button_init();

    if (!device_is_ready(dev)) {
        printk("BMI270 não está pronto\n");
        return 0;
    }

    /* ------ Configura acelerômetro ------ */
    full_scale_accel.val1 = 2;
    full_scale_accel.val2 = 0;
    sampling_freq.val1    = 100;
    sampling_freq.val2    = 0;
    oversampling.val1     = 1;
    oversampling.val2     = 0;

    sensor_attr_set(dev, SENSOR_CHAN_ACCEL_XYZ,
                    SENSOR_ATTR_FULL_SCALE, &full_scale_accel);
    sensor_attr_set(dev, SENSOR_CHAN_ACCEL_XYZ,
                    SENSOR_ATTR_OVERSAMPLING, &oversampling);
    sensor_attr_set(dev, SENSOR_CHAN_ACCEL_XYZ,
                    SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);

    /* ------ Configura giroscópio ------ */
    full_scale_gyro.val1 = 500;
    full_scale_gyro.val2 = 0;

    sensor_attr_set(dev, SENSOR_CHAN_GYRO_XYZ,
                    SENSOR_ATTR_FULL_SCALE, &full_scale_gyro);
    sensor_attr_set(dev, SENSOR_CHAN_GYRO_XYZ,
                    SENSOR_ATTR_OVERSAMPLING, &oversampling);
    sensor_attr_set(dev, SENSOR_CHAN_GYRO_XYZ,
                    SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);

    /* ------ Calibração ------ */
    calibrate_gyro(dev);

    printk("Pressione sw0 para iniciar.\n");

    /* Timestamp para dt real */
    int64_t last_ts = 0;

    /* ------ Loop principal ------ */
    while (1) {

        /* Bloqueia enquanto pausado */
        if (!atomic_get(&app_running)) {
            printk("Pausado\n");
            k_sem_take(&run_sem, K_FOREVER);
            printk("Retomado\n");
            last_ts = 0; /* evita salto no dt ao retomar */
        }

        /* Lê sensor */
        sensor_sample_fetch(dev);
        sensor_channel_get(dev, SENSOR_CHAN_ACCEL_XYZ, acc);
        sensor_channel_get(dev, SENSOR_CHAN_GYRO_XYZ,  gyr);

        /* Acelerômetro com filtro passa-baixa
         * Reduz ruído de alta frequência sem atrasar a resposta */
        float ax_raw = (float)sensor_value_to_double(&acc[0]);
        float ay_raw = (float)sensor_value_to_double(&acc[1]);
        float az_raw = (float)sensor_value_to_double(&acc[2]);

        ax_f = ACC_ALPHA * ax_raw + (1.0f - ACC_ALPHA) * ax_f;
        ay_f = ACC_ALPHA * ay_raw + (1.0f - ACC_ALPHA) * ay_f;
        az_f = ACC_ALPHA * az_raw + (1.0f - ACC_ALPHA) * az_f;

        /* Giroscópio com subtração de offset 
         * Sem deadband — o beta adaptativo resolve o drift
         * de forma mais elegante e sem congelar o filtro */
        float gx = (float)sensor_value_to_double(&gyr[0]) - gyr_offset_x;
        float gy = (float)sensor_value_to_double(&gyr[1]) - gyr_offset_y;
        float gz = (float)sensor_value_to_double(&gyr[2]) - gyr_offset_z;

        /* dt real */
        int64_t now = k_uptime_get();
        float dt = (last_ts == 0) ? 0.01f
                                  : (float)(now - last_ts) / 1000.0f;
        last_ts = now;

        /* Beta adaptativo sem congelamento do filtro
         *
         * A diferença crítica em relação ao seu código original:
         *   Antes: is_static → congela filterUpdate() completamente
         *   Agora: is_static → só aumenta o beta
         *
         * O filtro SEMPRE roda. Quando parado, o acelerômetro
         * tem mais peso e puxa o quaternion de volta à posição
         * correta — sem travar o erro acumulado no lugar.
         */
        bool static_now = is_sensor_static(gx, gy, gz);
        float beta = static_now ? BETA_STATIC : BETA_MOTION;

        /* Atualiza o filtro com acelerômetro filtrado */
        madgwick_update(gx, gy, gz, ax_f, ay_f, az_f, dt, beta);

        /* Envia quaternion pela serial
         * Protocolo compatível com o visualizador */
        printk("Q:%f,%f,%f,%f\n",
               (double)q0, (double)q1, (double)q2, (double)q3);

        k_sleep(K_MSEC(10));
    }
}
