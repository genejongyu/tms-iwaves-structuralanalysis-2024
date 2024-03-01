TITLE esser_mech_pp.mod

NEURON {
    SUFFIX esser_mech_pp
    POINT_PROCESS esser_mech_pp
    RANGE gNa_leak, gK_leak, theta_eq, tau_theta, tau_spike, tau_m, gspike, C, tspike, v_iaf, theta, v_iaf0, ena_iaf, ek_iaf, i_stim
    RANGE ina_iaf, ik_iaf, ispike, i_total, v_tms
    POINTER i_ampa, i_nmda, i_gabaa, i_gabab, i_noise
}
UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
}

PARAMETER {
    gNa_leak = 0.14
    gK_leak = 1.0
    C = 0.85
    theta_eq = -53
    tau_theta = 2
    tau_spike = 1.75
    tau_m = 15
    tspike = 2
    i_stim = 0
    v_tms = 0
}
ASSIGNED {
    ena_iaf
    ek_iaf
    i_ampa
    i_nmda
    i_gabaa
    i_gabab	
    i_noise
    i_total
    gspike
    vaux
}

STATE {
    theta
    v_iaf
    ina_iaf
    ik_iaf
    ispike
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina_iaf = gNa_leak*(v_iaf-ena_iaf)     : Sodium-like leak current
    ik_iaf = gK_leak*(v_iaf-ek_iaf)    : Potassium-like leak current
    ispike = gspike*(v_iaf-ek_iaf)
    i_total = ina_iaf+ik_iaf-i_stim+ispike+i_ampa+i_nmda+i_gabaa+i_gabab+i_noise
    v_iaf = v_iaf
    vaux = v_iaf-theta + v_tms
}

INITIAL {
    net_send(0, 1)
    theta = theta_eq
    gspike = 0
    ena_iaf = 30
    ek_iaf = -90
    v_iaf = v_iaf0
    ina_iaf = gNa_leak*(v_iaf0-ena_iaf)
    ik_iaf = gK_leak*(v_iaf0-ek_iaf)
    ispike = gspike*(v_iaf0-ek_iaf)
    vaux = v_iaf0-theta_eq
    i_ampa = 0
    i_nmda = 0
    i_gabaa = 0
    i_gabab = 0				 
}

DERIVATIVE states {
    theta' = (-(theta - theta_eq) + C*(v_iaf - theta_eq))/tau_theta  : threshold
    v_iaf' = -((ina_iaf+ik_iaf-i_stim+i_ampa+i_nmda+i_gabaa+i_gabab+i_noise)/tau_m + ispike/tau_spike)
}

NET_RECEIVE (w) {
    if (flag == 1){
        if (vaux > 0){
            theta = ena_iaf
            v_iaf = ena_iaf
        }
        WATCH (vaux > 0) 2
    }else if (flag == 2){
        net_event(t)
        gspike = 1
        theta = ena_iaf
        v_iaf = ena_iaf
        net_send(tspike, 3)
    }else if (flag == 3){
        gspike = 0
        net_send(0, 1)
    }
}
