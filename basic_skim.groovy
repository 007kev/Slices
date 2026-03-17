

// ======================================================
//  CLAS12 FILTERED ANALYSIS EXAMPLE
//  - Requires 1 electron + 2 K+
//  - theta between 5–35 degrees
//  - status between 2000–4000
//  - Missing mass of eK+K+
//  - Writes four-vectors + calorimeter info to text file
//  - Plots p, theta, and missing mass
// ======================================================

// -------- Imports --------
import org.jlab.jnp.hipo4.io.*
import org.jlab.jnp.hipo4.data.*
import org.jlab.groot.data.*
import org.jlab.groot.graphics.*
import org.jlab.groot.ui.*
import org.jlab.clas.physics.LorentzVector

import javax.swing.JFrame
import java.io.*
import java.util.ArrayList

// ======================================================
//  USER DIRECTORY
// ======================================================

String dir = "/Users/biancagualtieri/Desktop/researchFall25/testSkimHipo/"

// ======================================================
//  OUTPUT TEXT FILE
// ======================================================

PrintWriter writer = new PrintWriter(new FileWriter("Filtered_e_KK_events.txt"))

// ======================================================
//  HISTOGRAMS
// ======================================================

H1F h_el_p     = new H1F("Electron p", 100, 0, 8)
H1F h_el_theta = new H1F("Electron theta", 100, 0, 40)

H1F h_kp_p     = new H1F("K+ p", 100, 0, 5)
H1F h_kp_theta = new H1F("K+ theta", 100, 0, 40)

H1F h_mm       = new H1F("Missing Mass eK+K+", 120, 0.0, 3.0)

h_el_p.setTitleX("p (GeV)")
h_el_theta.setTitleX("theta (deg)")
h_kp_p.setTitleX("p (GeV)")
h_kp_theta.setTitleX("theta (deg)")
h_mm.setTitleX("Missing Mass (GeV)")

// ======================================================
//  PHYSICS CONSTANTS
// ======================================================

float melec = 0.000511
float mkaon = 0.49367
float mpro  = 0.93827

// 6.5 GeV electron beam
float beamEnergy = 6.5
LorentzVector beam   = new LorentzVector(0,0,beamEnergy,beamEnergy)
LorentzVector target = new LorentzVector(0,0,0,mpro)

// ======================================================
//  FILE LOOP
// ======================================================

File directory = new File(dir)
String[] filesList = directory.list()

for(int f = 0; f < filesList.length; f++) {

    println("Reading: " + filesList[f])

    HipoReader reader = new HipoReader()
    reader.open(dir + filesList[f])

    Event event = new Event()

    // ---- add in any bank you want to read from the hipo file here ---//
    Bank particles = new Bank(reader.getSchemaFactory().getSchema("REC::Particle"))
    Bank cal = new Bank(reader.getSchemaFactory().getSchema("REC::Calorimeter"))

    while(reader.hasNext()) {

        reader.nextEvent(event)
        event.read(particles)
        event.read(cal)

        ArrayList<Map> electrons = new ArrayList<>()
        ArrayList<Map> kaons = new ArrayList<>()

        int rows = particles.getRows()

        // ==================================================
        //  PARTICLE LOOP WITH CUTS
        // ==================================================
        // reading in the variables from the data banks //
        for(int i = 0; i < rows; i++) {

            int pid = particles.getInt("pid", i)
            int status = particles.getShort("status", i)

            float px = particles.getFloat("px", i)
            float py = particles.getFloat("py", i)
            float pz = particles.getFloat("pz", i)

            float p = (float)Math.sqrt(px*px + py*py + pz*pz)
            float theta = (float)Math.toDegrees(Math.acos(pz/p))

            // set some selection cuts here //
            if(theta < 5 || theta > 35) continue
            if(status < 2000 || status > 4000) continue

            // make your four-vectors here//
            if(pid == 11) {
                float E = (float)Math.sqrt(p*p + melec*melec)
                electrons.add([index:i, lv:new LorentzVector(px,py,pz,E), p:p, theta:theta])
            }

            if(pid == 321) {
                float E = (float)Math.sqrt(p*p + mkaon*mkaon)
                kaons.add([index:i, lv:new LorentzVector(px,py,pz,E), p:p, theta:theta])
            }
        }

        // ==================================================
        //  REQUIRE EXACTLY 1 e AND 2 K+
        // ==================================================

        if(electrons.size() == 1 && kaons.size() == 2) {

            def e  = electrons.get(0)
            def k1 = kaons.get(0)
            def k2 = kaons.get(1)

            // Fill histograms
            h_el_p.fill(e.p)
            h_el_theta.fill(e.theta)

            h_kp_p.fill(k1.p)
            h_kp_theta.fill(k1.theta)

            h_kp_p.fill(k2.p)
            h_kp_theta.fill(k2.theta)

            // ==================================================
            //  MISSING MASS CALCULATION
            // ==================================================

            LorentzVector X = new LorentzVector(beam)
            X.add(target)
            X.sub(e.lv)
            X.sub(k1.lv)
            X.sub(k2.lv)

            float mm = (float)X.mass()
            h_mm.fill(mm)

            // ==================================================
            //  WRITE EVENT INFO TO FILE
            // ==================================================

            writer.println("====================================")
            writer.println("Missing Mass = " + mm)

            writer.println("Electron: px py pz E = "
                    + e.lv.px() + " "
                    + e.lv.py() + " "
                    + e.lv.pz() + " "
                    + e.lv.e())

            writer.println("K1: px py pz E = "
                    + k1.lv.px() + " "
                    + k1.lv.py() + " "
                    + k1.lv.pz() + " "
                    + k1.lv.e())

            writer.println("K2: px py pz E = "
                    + k2.lv.px() + " "
                    + k2.lv.py() + " "
                    + k2.lv.pz() + " "
                    + k2.lv.e())
        }
    }

    reader.close()
}

writer.close()

// ======================================================
//  DRAW HISTOGRAMS
// ======================================================

JFrame frame = new JFrame("Filtered Distributions")
EmbeddedCanvas canvas = new EmbeddedCanvas()

canvas.divide(3,2)

canvas.cd(0); canvas.draw(h_el_p)
canvas.cd(1); canvas.draw(h_el_theta)
canvas.cd(2); canvas.draw(h_kp_p)
canvas.cd(3); canvas.draw(h_kp_theta)
canvas.cd(4); canvas.draw(h_mm)

frame.add(canvas)
frame.setSize(1200,800)
frame.setVisible(true)

println("Done. Missing mass plot added.")