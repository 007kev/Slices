#include <TTree.h>
#include <TFile.h>
#include <TDatabasePDG.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <TCanvas.h>
#include <TBenchmark.h>
#include <iostream>
#include <vector>
#include <cmath> // Required for std::acos and std::sqrt
#include "clas12reader.h"
// #include "HipoChain.h"


void SetLorentzVector(TLorentzVector &p4, clas12::region_part_ptr rp)
{
    p4.SetXYZM(rp->par()->getPx(), rp->par()->getPy(), rp->par()->getPz(), p4.M());
}

TLorentzVector CorrectElectron(TLorentzVector &p4) 
{
    Double_t E_cor, px_el, py_el, pz_el;
    TLorentzVector el_new;

    E_cor = p4.E() + 0.085643 - 0.0288063*p4.E() + 0.00894691*p4.E()*p4.E() - 0.000725449*p4.E()*p4.E()*p4.E();

    px_el = E_cor * (p4.Px() / p4.Rho());
    py_el = E_cor * (p4.Py() / p4.Rho());
    pz_el = E_cor * (p4.Pz() / p4.Rho());

    el_new.SetXYZM(px_el, py_el, pz_el, 0.000511);

    return el_new;
}

struct ParticleInfo
{
    int pid;
    int charge;
    float px;
    float py; 
    float pz; 
    float P_mag;
    float vx; 
    float vy; 
    float vz;
    float theta; 
    float phi;
    float deltaTime; 
    float beta;
    float betafromP;
    float path;
    int region; 
    int status;
    float chi2pid;
};

void getParticle(ParticleInfo& info, const clas12::region_part_ptr particle) 
{
    info.pid = particle->getPid();
    info.P_mag = particle->getP();
    info.px = particle->getPx();
    info.py = particle->getPy();
    info.pz = particle->getPz();
    info.vx = particle->par()->getVx();
    info.vy = particle->par()->getVy();
    info.vz = particle->par()->getVz();
    info.theta = particle->getTheta();
    info.phi = particle->getPhi();
    info.deltaTime = particle->getDeltaTime();
    info.beta = particle->getBeta();
    info.betafromP = particle->getBetaFromP();
    info.region = particle->getRegion();
    info.status = particle->getStatus();
    info.chi2pid = particle->getChi2Pid();
}

void writeParticleInfoToTree(ParticleInfo& info, TTree* tree, const std::string& suffix) 
{
    tree->Branch(("pid_" + suffix).c_str(), &info.pid);
    tree->Branch(("charge_" + suffix).c_str(), &info.charge);
    tree->Branch(("px_" + suffix).c_str(), &info.px);
    tree->Branch(("py_" + suffix).c_str(), &info.py);
    tree->Branch(("pz_" + suffix).c_str(), &info.pz);
    tree->Branch(("P_mag_" + suffix).c_str(), &info.P_mag);
    tree->Branch(("vx_" + suffix).c_str(), &info.vx);
    tree->Branch(("vy_" + suffix).c_str(), &info.vy);
    tree->Branch(("vz_" + suffix).c_str(), &info.vz);
    tree->Branch(("theta_" + suffix).c_str(), &info.theta);
    tree->Branch(("phi_" + suffix).c_str(), &info.phi);
    tree->Branch(("deltaTime_" + suffix).c_str(), &info.deltaTime);
    tree->Branch(("beta_" + suffix).c_str(), &info.beta);
    tree->Branch(("betafromP_" + suffix).c_str(), &info.betafromP);
    tree->Branch(("region_" + suffix).c_str(), &info.region);
    tree->Branch(("status_" + suffix).c_str(), &info.status);
    tree->Branch(("chi2pid_" + suffix).c_str(), &info.chi2pid);
}

void v4_hipo_root_pppim()
{
    // ---> NEW: Start the stopwatch
    gBenchmark->Start("conversion_timer");

    auto db = TDatabasePDG::Instance();
    
    // --- Event counters ---
    Long64_t n_total        = 0;  
    Long64_t n_have_topo    = 0;  

    Double_t mass_e = db->GetParticle(11)->Mass();
    Double_t mass_p = db->GetParticle(2212)->Mass();
    Double_t mass_pim = db->GetParticle(211)->Mass();   

    Double_t energy = 10.1998;  

    TLorentzVector beam(0, 0, sqrt(energy*energy - mass_e*mass_e), energy);
    TLorentzVector target(0, 0, 0, db->GetParticle(2212)->Mass());
    TLorentzVector p_electron(0, 0, 0, db->GetParticle(11)->Mass());
    TLorentzVector p_proton1(0, 0, 0, db->GetParticle(2212)->Mass());
    TLorentzVector p_proton2(0, 0, 0, db->GetParticle(2212)->Mass());
    TLorentzVector p_pim(0, 0, 0, db->GetParticle(211)->Mass());

    clas12root::HipoChain chain;
        
    auto config_c12 = chain.GetC12Reader();
    auto& c12 = chain.C12ref();
    c12->useFTBased();

    
    Double_t pp_inv_mass, miss_mass, miss_mass_sq;
    float e_status_val; // Branch variable to store the electron status

    // Hipo file to read
    chain.Add("/lustre24/expphy/volatile/clas12/leomart/Data/Runs/Spring2019/FT_merged/Pp_eFT_all.hipo");
    
    // Vectors to hold orphan hit information
    std::vector<float> orphan_E;
    std::vector<float> orphan_x;
    std::vector<float> orphan_y;
    std::vector<float> orphan_z;

    // Defining the root file and trees
    TFile *file = new TFile("v6_kev_Pppim_eFT_all.root","RECREATE");
    TTree *tree_indiv = new TTree("Individual", "Individual particle variables");

    tree_indiv->Branch("miss_mass", &miss_mass);
    tree_indiv->Branch("miss_mass_sq", &miss_mass_sq);
    tree_indiv->Branch("pp_inv_mass", &pp_inv_mass);
    
    // ---> NEW: Branch to save the electron status so Python can filter it!
    tree_indiv->Branch("e_status", &e_status_val, "e_status/F");

    ParticleInfo electronInfo, proton1Info, proton2Info, piminusInfo;
    
    writeParticleInfoToTree(electronInfo, tree_indiv, "e");
    writeParticleInfoToTree(proton1Info, tree_indiv, "p1");
    writeParticleInfoToTree(proton2Info, tree_indiv, "p2");
    writeParticleInfoToTree(piminusInfo, tree_indiv, "pim");

    tree_indiv->Branch("orphan_E", &orphan_E);
    tree_indiv->Branch("orphan_x", &orphan_x);
    tree_indiv->Branch("orphan_y", &orphan_y);
    tree_indiv->Branch("orphan_z", &orphan_z);


    while(chain.Next()){

        orphan_E.clear();
        orphan_x.clear();
        orphan_y.clear();
        orphan_z.clear();
        
        c12->event()->getStartTime();
        n_total++;  
        
        p_electron.SetXYZM(0, 0, 0, mass_e);
        p_proton1.SetXYZM(0, 0, 0, mass_p);
        p_proton2.SetXYZM(0, 0, 0, mass_p);
        p_pim.SetXYZM(0, 0, 0, mass_pim);
    
        miss_mass = -999;
        miss_mass_sq = -999;
        pp_inv_mass = -999;
        e_status_val = -999; 

        auto electrons = c12->getByID(11);
        auto protons = c12->getByID(2212);
        auto piminus = c12->getByID(-211);

        // 1. "At least" safety check to prevent crashes!
        // This explicitly requires AT LEAST 1 electron, 2 protons, and 1 pion.
        // if (electrons.size() < 1 || protons.size() < 2 || piminus.size() < 1) continue;

        // 2. The Combined Filter: Accept both FD (<0) and FT (1000-2000)
        int e_status = electrons[0]->getStatus();
        bool is_valid_electron = (e_status < 0 || (e_status >= 1000 && e_status < 2000));

        if (electrons[0]->getStatus() < 0) {
			if ((protons.size() >= 2) && (piminus.size() >= 1)) {
				// Set 4 vectors for first detected electron and first two protons //
				SetLorentzVector(p_electron, electrons[0]);
				SetLorentzVector(p_proton1, protons[0]);
				SetLorentzVector(p_proton2, protons[1]);
				SetLorentzVector(p_pim, piminus[0]);
				
				// p_electron_cor = CorrectElectron(p_electron);

				// Missing mass technique to find pbar //
				TLorentzVector MM = beam + target - p_electron - p_proton1 - p_proton2 - p_pim;
				miss_mass = MM.M();// Missing Mass
				miss_mass_sq = MM.M2();// Missing Mass squared
				pp_inv_mass = (p_proton1 + p_proton2).M();

				// Gets all wanted particle information	
				getParticle(electronInfo, electrons[0]);
				getParticle(proton1Info, protons[0]);
				getParticle(proton2Info, protons[1]);
				getParticle(piminusInfo, piminus[0]);
        
                // %%%%%%%%%%%%%%%%%%%%%%%% Calorimeter Block %%%%%%%%%%%%%%%%%%%%%%%%%%%
                auto &calos = c12->getRECCalorimeter();
                int ie = electrons[0]->getIndex();
                int ip1 = protons[0]->getIndex();
                int ip2 = protons[1]->getIndex();
                int ipi = piminus[0]->getIndex();

                float start_x = electronInfo.vx;
                float start_y = electronInfo.vy;
                float start_z = electronInfo.vz;

                for (int i = 0; i < calos.getRows(); i++) {
                    calos.setEntry(i);

                    int detector = calos.getDetector();  // 7 for ECAL system
                    int pindex = calos.getPindex();
                    
                    // Identify if this belongs to our primary 4 tracks
                    bool is_primary = (pindex == ie || pindex == ip1 || pindex == ip2 || pindex == ipi);

                    if (detector == 7 && !is_primary) {
                        float hit_x = calos.getX();
                        float hit_y = calos.getY();
                        float hit_z = calos.getZ();

                        float path_x = hit_x - start_x;
                        float path_y = hit_y - start_y;
                        float path_z = hit_z - start_z;

                        orphan_E.push_back(calos.getEnergy());
                        orphan_x.push_back(path_x);
                        orphan_y.push_back(path_y);
                        orphan_z.push_back(path_z);
                    }
                }
                // %%%%%%%%%%%%%%%%%%%%%%%% Calorimeter Block:End %%%%%%%%%%%%%%%%%%%%%%%

                tree_indiv->Fill(); 
                n_have_topo++;
            }
        }
    }

    std::cout << "Total events in HIPO:          " << n_total       << std::endl;
    std::cout << "Events with 1e2p1pi- topology: " << n_have_topo   << std::endl;

    tree_indiv->Write();
    file->Close();

    std::cout << "\n--- Execution Time ---" << std::endl;
    gBenchmark->Show("conversion_timer");
}
