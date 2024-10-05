import { Link } from "react-router-dom";
import { Content } from "../components/Content";
import { Nav } from "../components/Nav";
import { routes } from "../router";
import proofIcon from "../assets/proof.svg"
import { downloadVerifierKey } from "../api";
import { useState } from "react";

export const Verify = () => {

  const [proof, setProof] = useState<File | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const PROOF_EXT = ".pf"
    if (!event.target.files) return
    const file = event.target.files[0]
    if (!file) return
    if (!file.name.endsWith(PROOF_EXT)) return
    setProof(file)
  }

  return (
    <Content>
      <Nav title={routes.verify.title} />
      <div className="flex flex-col gap-3 items-center">
        <ul className="list-decimal">
          <li>
            Go to the&nbsp;
            <Link to={routes.records.path}>Records page</Link>
          </li>
          <li className="leading-6">
            <div className="flex gap-0.5">
              Grab the Proof of Computation by clicking the
              <img
                className="flex w-6 h-6"
                src={proofIcon}
                alt="proof-icon"
                title="Download proof of computation"
              />
              icon
            </div>
          </li>
          <li>
            You can also&nbsp;
            <Link to={""} onClick={downloadVerifierKey}>download</Link>
            &nbsp;the Verifier Key (VK) for the Public Setup model
            and verify it yourself
          </li>
          <li>Attach the files and verify if your prediction was computed by our Model</li>
        </ul>

        <div className="flex flex-col gap-1 items-center w-min">
          <input
            type="file"
            name="verify-input-proof"
            accept=".pf"
            onChange={handleFileChange}
          />
          <div>
            <button onClick={() => {}}>Verify</button>
          </div>
        </div>

      </div>
    </Content>
  )
}