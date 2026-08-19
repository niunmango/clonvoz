"""
Pruebas para el procesador de texto y adaptación fonética rioplatense.
"""

from src.preprocessing.text_processor import convert_to_rioplatense, segment_script


def test_convert_to_rioplatense_sheismo():
    """Prueba las transformaciones de 'll' y 'y' al sonido 'sh'."""
    text = "Yo me llamo Ramiro y voy a la playa a tomar mate bajo la lluvia."
    result = convert_to_rioplatense(text)

    # 'Yo' -> 'sho' (o 'Sho')
    assert "sho" in result.lower()
    # 'llamo' -> 'shamo'
    assert "shamo" in result
    # 'playa' -> 'plasha'
    assert "plasha" in result
    # 'lluvia' -> 'shuvia'
    assert "shuvia" in result
    # la conjunción 'y' aislada no debe alterarse
    assert " y voy " in result


def test_convert_to_rioplatense_preserves_loanwords():
    """Prueba que palabras como 'YouTube' mantengan su pronunciación."""
    text = "Seguinos en YouTube para ver los videos."
    result = convert_to_rioplatense(text)
    assert "YouTube" in result


def test_convert_to_rioplatense_aspirate_s():
    """Prueba la aspiración opcional de /s/ ante consonantes."""
    text = "estás en el mismo lugar"
    result = convert_to_rioplatense(text, aspirate_s=True)
    assert "ehtás" in result
    assert "mihmo" in result


def test_segment_script():
    """Prueba la segmentación correcta de párrafos."""
    script = """
    Primer párrafo introductorio al podcast de clonación de voz.

    Segundo párrafo con explicación detallada de la arquitectura VoxCPM2.


    Tercer párrafo final de despedida y llamado a la acción.
    """
    bloques = segment_script(script)
    assert len(bloques) == 3
    assert bloques[0].startswith("Primer párrafo")
    assert bloques[1].startswith("Segundo párrafo")
    assert bloques[2].startswith("Tercer párrafo")
