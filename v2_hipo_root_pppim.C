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
#include "HipoChain.h"

using namespace std;

void SetLorentzVector(TLorentzVector &p4, clas12::region_part_ptr rp)
{
    p4.SetXYZM(rp->par()->getPx(), rp->par()->getPy(), rp->par()->getPz(), p4.M());
}

TLorentzVector CorrectElectron(TLorentzVector &p4)
{
    Double_t E_cor, px_el, py_el, pz_el;
    TLorentzVector el_new;

    E_cor = p4.E()
          + 0.085643
          - 0.0288063 * p4.E()
          + 0.00894691 * p4.E() * p4.E()
          - 0.000725449 * p4.E() * p4.E() * p4.E();

    px_el = E_cor * (p4.Px() / p4.Rho());
    py_el = E_cor * (p4.Py() / p4.Rho());
    pz_el = E_cor * (p4.Pz() / p4.Rho());

    el_new.SetXYZM(px_el, py_el, pz_el, 0.000511);

    return el_new;
}

struct ParticleInfo {
    int   pid;
    int   charge;
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
    int   region;
    int   status;
    float chi2pid;
};

void getParticle(ParticleInfo &info, const clas12::region_part_ptr particle)
{
    info.pid        = particle->getPid();
    info.P_mag      = particle->getP();
    info.px         = particle->getPx();
    info.py         = particle->getPy();
    info.pz         = particle->getPz();
    info.vx         = particle->par()->getVx();
    info.vy         = particle->par()->getVy();
    info.vz         = particle->par()->getVz();
    info.theta      = particle->getTheta();
    info.phi        = particle->getPhi();
    info.deltaTime  = particle->getDeltaTime();
    info.beta       = particle->getBeta();
    info.betafromP  = particle->getBetaFromP();
    info.region     = particle->getRegion();
    info.status     = particle->getStatus();
    info.chi2pid    = particle->getChi2Pid();
    info.charge     = particle->par()->getCharge();  // <-- ADDED
}

void writeParticleInfoToTree(ParticleInfo &info, TTree *tree, const std::string &suffix)
{
    tree->Branch(("pid_"        + suffix).c_str(), &info.pid);
    tree->Branch(("charge_"     + suffix).c_str(), &info.charge);
    tree->Branch(("px_"         + suffix).c_str(), &info.px);
    tree->Branch(("py_"         + suffix).c_str(), &info.py);
    tree->Branch(("pz_"         + suffix).c_str(), &info.pz);
    tree->Branch(("P_mag_"      + suffix).c_str(), &info.P_mag);
    tree->Branch(("vx_"         + suffix).c_str(), &info.vx);
    tree->Branch(("vy_"         + suffix).c_str(), &info.vy);
    tree->Branch(("vz_"         + suffix).c_str(), &info.vz);
    tree->Branch(("theta_"      + suffix).c_str(), &info.theta);
    tree->Branch(("phi_"        + suffix).c_str(), &info.phi);
    tree->Branch(("deltaTime_"  + suffix).c_str(), &info.deltaTime);
    tree->Branch(("beta_"       + suffix).c_str(), &info.beta);
    tree->Branch(("betafromP_"  + suffix).c_str(), &info.betafromP);
    tree->Branch(("region_"     + suffix).c_str(), &info.region);
    tree->Branch(("status_"     + suffix).c_str(), &info.status);
    tree->Branch(("chi2pid_"    + suffix).c_str(), &info.chi2pid);
}

void v2_hipo_root_pppim()
{

    // ---> NEW: Start the stopwatch
    gBenchmark->Start("conversion_timer");

    auto db = TDatabasePDG::Instance();

    // --- Event counters ---
    Long64_t n_total        = 0;

    auto db = TDatabasePDG::Instance();

    // --- Event counters ---
    Long64_t n_total        = 0;  // all events in the HIPO file
    Long64_t n_have_topo    = 0;  // events with 1e, 2p, 1π-
    Long64_t n_status_ok    = 0;  // plus electron status < 0
    Long64_t n_filled_tree  = 0;  // events that actually go to the TTree
    Long64_t n_has_neutral  = 0;  // events where has_neutral == 1
    // -----------------------

    Double_t mass_e   = db->GetParticle(11)->Mass();
    Double_t mass_p   = db->GetParticle(2212)->Mass();
    Double_t mass_pim = db->GetParticle(211)->Mass();

    Double_t energy = 10.1998;

    TLorentzVector beam  (0, 0, sqrt(energy * energy - mass_e * mass_e), energy);
    TLorentzVector target(0, 0, 0, db->GetParticle(2212)->Mass());
    TLorentzVector p_electron(0, 0, 0, db->GetParticle(11)->Mass());
    TLorentzVector p_proton1 (0, 0, 0, db->GetParticle(2212)->Mass());
    TLorentzVector p_proton2 (0, 0, 0, db->GetParticle(2212)->Mass());
    TLorentzVector p_pim     (0, 0, 0, db->GetParticle(211)->Mass());

    clas12root::HipoChain chain;
    auto  config_c12 = chain.GetC12Reader();
    auto &c12        = chain.C12ref();

    chain.Add("/lustre24/expphy/volatile/clas12/leomart/Data/Runs/Spring2019/FT_merged/Pp_eFT_all.hipo");

    Double_t pp_inv_mass, miss_mass, miss_mass_sq;
    TLorentzVector p_electron_cor;

    float Ecal_e  = 0, Pcal_e  = 0;
    float Ecal_p1 = 0, Ecal_p2 = 0;
    float Ecal_pim = 0;

    float Enbar_calo  = 0.0f;
	float neutral_angle = -999.0f;
    float neutral_phi = -999.0f;
    int   has_neutral = 0;

    // ---> NEW: Dynamic lists for all orphaned hits
    std::vector<float> orphan_E;
    std::vector<float> orphan_angle;
    std::vector<float> orphan_phi;

	float e_status_val = 0;

    TFile *file       = new TFile("v2_kev_Pppim_eFT_all.root", "RECREATE");
    TTree *tree_indiv = new TTree("Individual", "Individual particle variables");

    tree_indiv->Branch("miss_mass",     &miss_mass);
    tree_indiv->Branch("miss_mass_sq",  &miss_mass_sq);
    tree_indiv->Branch("pp_inv_mass",   &pp_inv_mass);

    ParticleInfo electronInfo, proton1Info, proton2Info, piminusInfo;

    writeParticleInfoToTree(electronInfo, tree_indiv, "e");
    writeParticleInfoToTree(proton1Info,  tree_indiv, "p1");
    writeParticleInfoToTree(proton2Info,  tree_indiv, "p2");
    writeParticleInfoToTree(piminusInfo,  tree_indiv, "pim");

    tree_indiv->Branch("Ecal_e",   &Ecal_e,   "Ecal_e/F");
    tree_indiv->Branch("Pcal_e",   &Pcal_e,   "Pcal_e/F");
    tree_indiv->Branch("Ecal_p1",  &Ecal_p1,  "Ecal_p1/F");
    tree_indiv->Branch("Ecal_p2",  &Ecal_p2,  "Ecal_p2/F");
    tree_indiv->Branch("Ecal_pim", &Ecal_pim, "Ecal_pim/F");

	tree_indiv->Branch("e_status", &e_status_val, "e_status/F");

    tree_indiv->Branch("Enbar_calo",  &Enbar_calo,  "Enbar_calo/F");
	tree_indiv->Branch("neutral_angle", &neutral_angle, "neutral_angle/F");
    tree_indiv->Branch("has_neutral", &has_neutral, "has_neutral/I");
    tree_indiv->Branch("neutral_phi", &neutral_phi, "neutral_phi/F");

    // ---> NEW: Create branches for the dynamic lists
    tree_indiv->Branch("orphan_E", &orphan_E);
    tree_indiv->Branch("orphan_angle", &orphan_angle);
    tree_indiv->Branch("orphan_phi", &orphan_phi);

    while (chain.Next()) {

        // ---> NEW: Empty the lists for the new event
        orphan_E.clear();
        orphan_angle.clear();

        c12->event()->getStartTime();

        n_total++;  // +1 every event

        p_electron.SetXYZM(0, 0, 0, mass_e);
        p_proton1.SetXYZM (0, 0, 0, mass_p);
        p_proton2.SetXYZM (0, 0, 0, mass_p);
        p_pim.SetXYZM     (0, 0, 0, mass_pim);

        miss_mass    = -999;
        miss_mass_sq = -999;
        pp_inv_mass  = -999;

        auto electrons = c12->getByID(11);
        auto protons   = c12->getByID(2212);
        auto piminus   = c12->getByID(-211);

		// --- 1. Basic Topology Check ---
        if (electrons.size() < 1 || protons.size() < 2 || piminus.size() < 1) continue;

        n_have_topo++;  // has 1e, 2p, 1π-

		// --- 2. THE ELECTRON SELECTION CHANGE ---
        int e_status = electrons[0]->getStatus();

		// We accept FD (status < 0) and FT (status 1000-2000)
        // std::abs() is a safe way to check the magnitude
        bool is_valid_electron = (std::abs(e_status) >= 1000 && std::abs(e_status) < 4000);

		if (is_valid_electron) {
            if (protons.size() == 2 && piminus.size() == 1) {

                n_status_ok++;  // passes status/topology

                SetLorentzVector(p_electron, electrons[0]);
                SetLorentzVector(p_proton1,  protons[0]);
                SetLorentzVector(p_proton2,  protons[1]);
                SetLorentzVector(p_pim,      piminus[0]);
				
				e_status_val = (float)electrons[0]->getStatus();

				// --- 3. THE MOMENTUM CORRECTION CHANGE ---
                TLorentzVector p_electron_final;
                if (e_status < 0) {
                    // It's in the Forward Detector, apply your correction
                    p_electron_final = CorrectElectron(p_electron);
                } else {
                    // It's in the Forward Tagger, use raw momentum
                    p_electron_final = p_electron;
                }

                TLorentzVector MM = beam + target - p_electron_final - p_proton1 - p_proton2 - p_pim;
                miss_mass    = MM.M();
                miss_mass_sq = MM.M2();
                pp_inv_mass  = (p_proton1 + p_proton2).M();

                // %%%%%%%%%%%%%%%%%%%%%%%% Calorimeter Block %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                auto &calos = c12->getRECCalorimeter();

                int ie  = electrons[0]->getIndex();
                int ip1 = protons[0]->getIndex();
                int ip2 = protons[1]->getIndex();
                int ipi = piminus[0]->getIndex();

                Ecal_e  = Pcal_e  = 0.0f;
                Ecal_p1 = Ecal_p2 = 0.0f;
                Ecal_pim = 0.0f;
                Enbar_calo  = 0.0f;
                neutral_angle = -999.0f;
                has_neutral = 0;

				Enbar_calo = 0.0f;
				neutral_angle = -999.0f;
                neutral_phi = -999.0f;
				has_neutral = 0;

                TVector3 p_miss = MM.Vect();
                double   best_angle = 1e9;

                for (int i = 0; i < calos.getRows(); i++) {
                    calos.setEntry(i);

                    int   detector = calos.getDetector();  // 7 for ECAL system
                    int   pindex   = calos.getPindex();
                    int   layer    = calos.getLayer();
                    float Ehit     = calos.getEnergy();
                    float x        = calos.getX();
                    float y        = calos.getY();
                    float z        = calos.getZ();			
                    
					if (detector != 7) continue;

                    if (pindex == ie) {
                        if (layer == 1)      Pcal_e  += Ehit;
                        else if (layer >= 4) Ecal_e  += Ehit;
                    }
                    else if (pindex == ip1 && layer >= 4) Ecal_p1 += Ehit;
                    else if (pindex == ip2 && layer >= 4) Ecal_p2 += Ehit;
                    else if (pindex == ipi && layer >= 4) Ecal_pim += Ehit;

					// Identify if this belongs to our primary 4 tracks
    				bool is_primary = (pindex == ie || pindex == ip1 || pindex == ip2 || pindex == ipi);

                    if (!is_primary) {
                        TVector3 r_hit(x, y, z);
                        // double angle = r_hit.Angle(p_miss); // used to calculate the angle between the hit and the missing momentum vector
                        // lets do it manually
                        double dot_product = (r_hit.X() * p_miss.X()) + (r_hit.Y() * p_miss.Y()) + (r_hit.Z() * p_miss.Z());
                        double mag_r_hit = std::sqrt( r_hit.X() * r_hit.X() + r_hit.Y() * r_hit.Y() +r_hit.Z() * r_hit.Z() );
                        double mag_p_miss = std::sqrt( p_miss.X() * p_miss.X() + p_miss.Y() * p_miss.Y() + p_miss.Z() * p_miss.Z() );

                        // to protect against numbers outside of range of cosine that would produce Nan
                        // I clamp it between -1 and 1
                        double angle = 0.0;
                        if (mag_r_hit > 0 && mag_p_miss > 0) {
                            double ratio = dot_product / (mag_r_hit * mag_p_miss);
                            if (ratio > 1.0) ratio = 1.0;
                            if (ratio < -1.0) ratio = -1.0;
                            angle = std::acos(ratio);

                            // now for azimuthal direction
                            double hit_phi = std::atan2(r_hit.Y(), r_hit.X());
                            double p_miss_phi = std::atan2(p_miss.Y(), p_miss.X());
                            double delta_phi = hit_phi - p_miss_phi;
                            
                            // Wrap delta_phi to the range [-pi, pi]
                            if (delta_phi > M_PI) delta_phi -= 2 * M_PI;
                            if (delta_phi < -M_PI) delta_phi += 2 * M_PI;

                            // ---> NEW: Add EVERY orphaned hit to your lists
                            orphan_E.push_back(Ehit);
                            orphan_angle.push_back((float)angle);
                            orphan_phi.push_back((float)delta_phi); // store the relative phi angle to the missing momentum

                            if (angle < best_angle) {
                                best_angle  = angle;
                                Enbar_calo  = Ehit;
                                neutral_angle = (float)angle;
                                neutral_phi = (float)delta_phi; // store the relative phi angle of the closest hit
                                has_neutral = 1;
                            }
                        }
                    }
                }
                // %%%%%%%%%%%%%%%%%%%%%%%% Calorimeter Block:End %%%%%%%%%%%%%%%%%%%%%%%%%

                if (has_neutral == 1) {
                    n_has_neutral++;

                    getParticle(electronInfo, electrons[0]);
                    getParticle(proton1Info,  protons[0]);
                    getParticle(proton2Info,  protons[1]);
                    getParticle(piminusInfo,  piminus[0]);

                    n_filled_tree++;
                    tree_indiv->Fill();
                }
            }
        } // Closes the 'if' statements
    } // Closes the main 'while (chain.Next())' loop

    std::cout << "Total events in HIPO:          " << n_total       << std::endl;
    std::cout << "Events with 1e2p1pi- topology: " << n_have_topo   << std::endl;
    std::cout << "Events with status OK:         " << n_status_ok   << std::endl;
    std::cout << "Events written to TTree:       " << n_filled_tree << std::endl;
    std::cout << "Events with has_neutral == 1:  " << n_has_neutral << std::endl;

    tree_indiv->Write();
    file->Close();

    // ---> NEW: Stop the stopwatch and print the elapsed time
    std::cout << "\n--- Execution Time ---" << std::endl;
    gBenchmark->Show("conversion_timer");
}